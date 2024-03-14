"""
Cleanlab TLM is a Large Language Model that gives more reliable answers and quantifies its uncertainty in these answers
"""

from __future__ import annotations

import asyncio
import sys
from typing import Coroutine, List, Optional, Union, cast, Sequence
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

import aiohttp
from typing_extensions import NotRequired, TypedDict  # for Python <3.11 with (Not)Required

from cleanlab_studio.internal.api import api
from cleanlab_studio.internal.tlm_helpers import (
    validate_tlm_prompt,
    validate_tlm_try_prompt,
    validate_tlm_prompt_response,
    validate_try_tlm_prompt_response,
    validate_tlm_options,
)
from cleanlab_studio.internal.types import TLMQualityPreset
from cleanlab_studio.errors import ValidationError
from cleanlab_studio.internal.constants import (
    _DEFAULT_MAX_CONCURRENT_TLM_REQUESTS,
    _VALID_TLM_QUALITY_PRESETS,
    _TLM_MAX_RETRIES,
)


class TLMResponse(TypedDict):
    """Trustworthy Language Model response.

    Attributes:
        response (str): text response from language model
        trustworthiness_score (float): score corresponding to confidence that the response is correct
    """

    response: str
    trustworthiness_score: Optional[float]


class TLMOptions(TypedDict):
    """Trustworthy language model options. The TLM quality-preset determines many of these settings automatically, but
    specifying other values here will over-ride the setting from the quality-preset.

    Args:
        max_tokens (int, default = 512): the maximum number of tokens to generate in the TLM response.

        model (str, default = "gpt-3.5-turbo-16k"): ID of the model to use. Other options: "gpt-4"

        num_candidate_responses (int, default = 1): this controls how many candidate responses are internally generated.
        TLM scores the confidence of each candidate response, and then returns the most confident one.
        A higher value here can produce better (more accurate) responses from the TLM, but at higher costs/runtimes.

        num_consistency_samples (int, default = 5): this controls how many samples are internally generated to evaluate the LLM-response-consistency.
        This is a big part of the returned trustworthiness_score, in particular for ensuring lower scores for strange input prompts or those that are too open-ended to receive a well-defined 'good' response.
        Higher values here produce better (more reliable) TLM confidence scores, but at higher costs/runtimes.

        use_self_reflection (bool, default = `True`): this controls whether self-reflection is used to have the LLM reflect upon the response it is generating and explicitly self-evaluate whether it seems good or not.
        This is a big part of the confidence score, in particular for ensure low scores for responses that are obviously incorrect/bad for a standard prompt that LLMs should be able to handle.
        Setting this to False disables the use of self-reflection and may produce worse TLM confidence scores, but can reduce costs/runtimes.
    """

    max_tokens: NotRequired[int]
    model: NotRequired[str]
    num_candidate_responses: NotRequired[int]
    num_consistency_samples: NotRequired[int]
    use_self_reflection: NotRequired[bool]


BatchPromptResponse = List[TLMResponse]
TryBatchPromptResponse = List[Optional[TLMResponse]]
BatchGetTrustworthinessScoreResponse = List[float]
TryBatchGetTrustworthinessScoreResponse = List[Optional[float]]


class TLM:
    """TLM interface class."""

    def __init__(
        self,
        api_key: str,
        quality_preset: TLMQualityPreset,
        *,
        options: Optional[TLMOptions] = None,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = None,
    ) -> None:
        """Initializes TLM interface.

        Args:
            api_key (str): API key used to authenticate TLM client
            quality_preset (TLMQualityPreset): quality preset to use for TLM queries
            options (TLMOptions, optional): dictionary of options to pass to prompt method, defaults to None
            timeout (float, optional): timeout (in seconds) to run all prompts, defaults to None
            verbose (bool, optional): verbosity level for TLM queries, default to True which will print progress bars for TLM queries. For silent TLM progress, set to False.
        """
        self._api_key = api_key

        if quality_preset not in _VALID_TLM_QUALITY_PRESETS:
            raise ValidationError(
                f"Invalid quality preset {quality_preset} -- must be one of {_VALID_TLM_QUALITY_PRESETS}"
            )

        if options is not None:
            validate_tlm_options(options)

        if timeout is not None and not (isinstance(timeout, int) or isinstance(timeout, float)):
            raise ValidationError("timeout must be a integer or float value")

        if verbose is not None and not isinstance(verbose, bool):
            raise ValidationError("timeout must be a boolean value")

        is_notebook_flag = is_notebook()

        self._quality_preset = quality_preset
        self._options = options
        self._timeout = timeout if timeout is not None and timeout > 0 else None
        self._verbose = verbose if verbose is not None else is_notebook_flag

        if is_notebook_flag:
            import nest_asyncio

            nest_asyncio.apply()

        # TODO: figure out this how to compute appropriate max_concurrent_requests
        max_concurrent_requests = _DEFAULT_MAX_CONCURRENT_TLM_REQUESTS

        self._event_loop = asyncio.get_event_loop()
        self._query_semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def _batch_prompt(
        self,
        prompts: Sequence[str],
        capture_exceptions: bool = False,
    ) -> Union[BatchPromptResponse, TryBatchPromptResponse]:
        """Run batch of TLM prompts. The list returned will have the same length as the input list,

        If capture_exceptions is True, the list will contain None in place of the response for any errors or timeout processing some inputs.
        Otherwise, the method will raise an exception for any errors or timeout processing some inputs.

        Args:
            prompts (List[str]): list of prompts to run
            capture_exceptions (bool): if should return None in place of the response for any errors or timeout processing some inputs

        Returns:
            Union[BatchPromptResponse, TryBatchPromptResponse]: TLM responses for each prompt (in supplied order)
        """
        if capture_exceptions:
            per_query_timeout, per_batch_timeout = self._timeout, None
        else:
            per_query_timeout, per_batch_timeout = None, self._timeout

        # run batch of TLM
        tlm_responses = await self._batch_async(
            [
                self._prompt_async(
                    prompt, timeout=per_query_timeout, capture_exceptions=capture_exceptions
                )
                for prompt in prompts
            ],
            per_batch_timeout,
        )

        if capture_exceptions:
            return cast(TryBatchPromptResponse, tlm_responses)

        return cast(BatchPromptResponse, tlm_responses)

    async def _batch_get_trustworthiness_score(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        capture_exceptions: bool = False,
    ) -> Union[BatchGetTrustworthinessScoreResponse, TryBatchGetTrustworthinessScoreResponse]:
        """Run batch of TLM get confidence score.

        capture_exceptions behavior:
        - If true, the list will contain None in place of the response for any errors or timeout processing some inputs.
        - Otherwise, the method will raise an exception for any errors or timeout processing some inputs.

        capture_exceptions interaction with timeout:
        - If true, timeouts are applied on a per-query basis (i.e. some queries may succeed while others fail)
        - If false, a single timeout is applied to the entire batch (i.e. all queries will fail if the timeout is reached)

        Args:
            prompts (Sequence[str]): list of prompts to run get confidence score for
            responses (Sequence[str]): list of responses to run get confidence score for
            capture_exceptions (bool): if should return None in place of the response for any errors or timeout processing some inputs

        Returns:
            Union[List[float], List[Optional[float]]: TLM confidence score for each prompt (in supplied order)
        """
        if capture_exceptions:
            per_query_timeout, per_batch_timeout = self._timeout, None
        else:
            per_query_timeout, per_batch_timeout = None, self._timeout

        # run batch of TLM get confidence score
        tlm_responses = await self._batch_async(
            [
                self._get_trustworthiness_score_async(
                    prompt,
                    response,
                    timeout=per_query_timeout,
                    capture_exceptions=capture_exceptions,
                )
                for prompt, response in zip(prompts, responses)
            ],
            per_batch_timeout,
        )

        if capture_exceptions:
            return cast(TryBatchGetTrustworthinessScoreResponse, tlm_responses)

        return cast(BatchGetTrustworthinessScoreResponse, tlm_responses)

    async def _batch_async(
        self,
        tlm_coroutines: List[Coroutine[None, None, Union[TLMResponse, float, None]]],
        batch_timeout: Optional[float] = None,
    ) -> Union[List[TLMResponse], List[float], List[None]]:
        """Runs batch of TLM queries.

        Args:
            tlm_coroutines (List[Coroutine[None, None, Union[TLMResponse, float, None]]]): list of query coroutines to run, returning TLM responses or confidence scores (or None if capture_exceptions is True)
            batch_timeout (Optional[float], optional): timeout (in seconds) to run all queries, defaults to None (no timeout)

        Returns:
            Union[List[TLMResponse], List[float], List[None]]: list of coroutine results, with preserved order
        """
        tlm_query_tasks = [asyncio.create_task(tlm_coro) for tlm_coro in tlm_coroutines]

        if self._verbose:
            gather_task = (
                tqdm_asyncio.gather(
                    *tlm_query_tasks,
                    total=len(tlm_query_tasks),
                    desc="Querying TLM...",
                    bar_format="{desc} {percentage:3.0f}%|{bar}|",
                ),
            )
        else:
            gather_task = asyncio.gather(*tlm_query_tasks)

        return await asyncio.wait_for(gather_task, timeout=batch_timeout)  # type: ignore[arg-type]

    def prompt(
        self,
        prompt: Union[str, Sequence[str]],
        /,
    ) -> Union[TLMResponse, List[TLMResponse]]:
        """
        Get response and trustworthiness score from TLM.

        Args:
            prompt (str | Sequence[str]): prompt (or list of multiple prompts) for the TLM
        Returns:
            TLMResponse | List[TLMResponse]: [TLMResponse](#class-tlmresponse) object containing the response and trustworthiness score.
                    If multiple prompts were provided in a list, then a list of such objects is returned, one for each prompt.
        """
        validate_tlm_prompt(prompt)

        if isinstance(prompt, str):
            return cast(
                TLMResponse,
                self._event_loop.run_until_complete(
                    self._prompt_async(prompt, timeout=self._timeout, capture_exceptions=False),
                ),
            )

        return cast(
            List[TLMResponse],
            self._event_loop.run_until_complete(
                self._batch_prompt(prompt, capture_exceptions=False),
            ),
        )

    def try_prompt(
        self,
        prompt: Sequence[str],
        /,
    ) -> List[Optional[TLMResponse]]:
        """
        Get response and trustworthiness score from TLM.
        The list returned will have the same length as the input list, if there are any
        failures (errors or timeout) processing some inputs, the list will contain None
        in place of the response.

        Args:
            prompt (Sequence[str]): list of multiple prompts for the TLM
        Returns:
            List[Optional[TLMResponse]]: list of [TLMResponse](#class-tlmresponse) objects containing the response and trustworthiness score.
                Entries of the list will be None for prompts that fail (due to any errors or timeout).
        """
        validate_tlm_try_prompt(prompt)

        return cast(
            List[Optional[TLMResponse]],
            self._event_loop.run_until_complete(
                self._batch_prompt(prompt, capture_exceptions=True),
            ),
        )

    async def prompt_async(
        self,
        prompt: Union[str, Sequence[str]],
        /,
    ) -> Union[TLMResponse, List[TLMResponse]]:
        """
        (Asynchronously) Get response and trustworthiness score from TLM.

        Args:
            prompt (str | Sequence[str]): prompt (or list of multiple prompts) for the TLM
        Returns:
            TLMResponse | List[TLMResponse]: [TLMResponse](#class-tlmresponse) object containing the response and trustworthiness score.
                    If multiple prompts were provided in a list, then a list of such objects is returned, one for each prompt.
        """
        validate_tlm_prompt(prompt)

        async with aiohttp.ClientSession() as session:
            if isinstance(prompt, str):
                tlm_response = await self._prompt_async(
                    prompt, session, timeout=self._timeout, capture_exceptions=False
                )
                return cast(TLMResponse, tlm_response)

            return cast(
                List[TLMResponse],
                await self._batch_prompt(prompt, capture_exceptions=False),
            )

    async def _prompt_async(
        self,
        prompt: str,
        client_session: Optional[aiohttp.ClientSession] = None,
        timeout: Optional[float] = None,
        capture_exceptions: bool = False,
    ) -> Optional[TLMResponse]:
        """
        Private asynchronous method to get response and trustworthiness score from TLM.

        Args:
            prompt (str): prompt for the TLM
            # TODO -- document parameters
        Returns:
            TLMResponse: [TLMResponse](#class-tlmresponse) object containing the response and trustworthiness score
        """

        async with self._query_semaphore:
            tlm_response = await api.tlm_prompt(
                self._api_key,
                prompt,
                self._quality_preset,
                self._options,
                client_session,
                retries=_TLM_MAX_RETRIES,
            )
            try:
                tlm_response = await asyncio.wait_for(
                    api.tlm_prompt(
                        self._api_key,
                        prompt,
                        self._quality_preset,
                        self._options,
                        client_session,
                        retries=_TLM_MAX_RETRIES,
                    ),
                    timeout=timeout,
                )
            except Exception as e:
                if capture_exceptions:
                    return None
                raise e

        return {
            "response": tlm_response["response"],
            "trustworthiness_score": tlm_response["confidence_score"],
        }

    def get_trustworthiness_score(
        self,
        prompt: Union[str, Sequence[str]],
        response: Union[str, Sequence[str]],
    ) -> Union[float, List[float]]:
        """Gets trustworthiness score for prompt-response pair(s).
        The list returned will have the same length as the input list, if there are any
        failures (errors or timeout) processing some inputs, the list will contain None
        in place of the response.

        Args:
            prompt (str | Sequence[str]): prompt (or list of multiple prompts) for the TLM
            response (str | Sequence[str]): response (or list of multiple responses) for the TLM to evaluate
        Returns:
            float (or list of floats if multiple prompt-responses were provided) corresponding to the TLM's trustworthiness score.
                    The score quantifies how confident TLM is that the given response is good for the given prompt.
                    Entries of the list will be None for prompts that fail (due to any errors or timeout).
        """
        validate_tlm_prompt_response(prompt, response)

        if isinstance(prompt, str) and isinstance(response, str):
            return cast(
                float,
                self._event_loop.run_until_complete(
                    self._get_trustworthiness_score_async(
                        prompt, response, timeout=self._timeout, capture_exceptions=False
                    )
                ),
            )

        return cast(
            List[float],
            self._event_loop.run_until_complete(
                self._batch_get_trustworthiness_score(prompt, response, capture_exceptions=False)
            ),
        )

    def try_get_trustworthiness_score(
        self,
        prompt: Sequence[str],
        response: Sequence[str],
    ) -> List[Optional[float]]:
        """Gets trustworthiness score for prompt-response pair(s).

        Args:
            prompt (str | Sequence[str]): prompt (or list of multiple prompts) for the TLM
            response (str | Sequence[str]): response (or list of multiple responses) for the TLM to evaluate
        Returns:
            list of floats if multiple prompt-responses were provided corresponding to the TLM's trustworthiness score.
                    The score quantifies how confident TLM is that the given response is good for the given prompt.
        """
        validate_try_tlm_prompt_response(prompt, response)

        return cast(
            List[Optional[float]],
            self._event_loop.run_until_complete(
                self._batch_get_trustworthiness_score(prompt, response, capture_exceptions=True)
            ),
        )

    async def get_trustworthiness_score_async(
        self,
        prompt: Union[str, Sequence[str]],
        response: Union[str, Sequence[str]],
    ) -> Union[float, List[float]]:
        """(Asynchronously) gets trustworthiness score for prompt-response pair.

        Args:
            prompt (str | Sequence[str]): prompt (or list of multiple prompts) for the TLM
            response (str | Sequence[str]): response (or list of multiple responses) for the TLM to evaluate
        Returns:
            float (or list of floats if multiple prompt-responses were provided) corresponding to the TLM's trustworthiness score.
                    The score quantifies how confident TLM is that the given response is good for the given prompt.
        """
        validate_tlm_prompt_response(prompt, response)

        async with aiohttp.ClientSession() as session:
            if isinstance(prompt, str) and isinstance(response, str):
                trustworthiness_score = await self._get_trustworthiness_score_async(
                    prompt, response, session, timeout=self._timeout, capture_exceptions=False
                )
                return cast(float, trustworthiness_score)

            return cast(
                List[float],
                await self._batch_get_trustworthiness_score(
                    prompt, response, capture_exceptions=False
                ),
            )

    async def _get_trustworthiness_score_async(
        self,
        prompt: str,
        response: str,
        client_session: Optional[aiohttp.ClientSession] = None,
        timeout: Optional[float] = None,
        capture_exceptions: bool = False,
    ) -> Optional[float]:
        """Private asynchronous method to get trustworthiness score for prompt-response pair.

        Args:
            prompt: prompt for the TLM
            response: response for the TLM to evaluate
            client_session: async HTTP session to use for TLM query. Defaults to None.
            timeout: timeout (in seconds) to run the prompt, defaults to None (no timeout)
            capture_exceptions: if should return None in place of the response for any errors
        Returns:
            float corresponding to the TLM's trustworthiness score

        """
        if self._quality_preset == "base":
            raise ValueError(
                "Cannot get confidence score with `base` quality_preset -- choose a higher preset."
            )

        async with self._query_semaphore:
            try:
                return cast(
                    float,
                    (
                        await asyncio.wait_for(
                            api.tlm_get_confidence_score(
                                self._api_key,
                                prompt,
                                response,
                                self._quality_preset,
                                self._options,
                                client_session,
                                retries=_TLM_MAX_RETRIES,
                            ),
                            timeout=timeout,
                        )
                    )["confidence_score"],
                )
            except Exception as e:
                if capture_exceptions:
                    return None
                raise e


def is_notebook() -> bool:
    """Returns True if running in a notebook, False otherwise.

    lazydocs: ignore
    """
    try:
        get_ipython = sys.modules["IPython"].get_ipython
        if "IPKernelApp" in get_ipython().config:
            return True

        return False
    except:
        return False
