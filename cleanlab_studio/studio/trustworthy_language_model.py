"""
Cleanlab TLM is a Large Language Model that gives more reliable answers and quantifies its uncertainty in these answers
"""

from __future__ import annotations

import asyncio
import sys
from typing import Coroutine, List, Optional, Union, cast, Sequence
from tqdm.asyncio import tqdm_asyncio

import aiohttp
from typing_extensions import NotRequired, TypedDict  # for Python <3.11 with (Not)Required

from cleanlab_studio.internal.api import api
from cleanlab_studio.internal.tlm_helpers import (
    validate_tlm_prompt,
    validate_tlm_prompt_response,
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

    def _batch_prompt(
        self,
        prompts: Sequence[str],
    ) -> List[TLMResponse]:
        """Run batch of TLM prompts.

        Args:
            prompts (List[str]): list of prompts to run

        Returns:
            List[TLMResponse]: TLM responses for each prompt (in supplied order)
        """
        tlm_responses = self._event_loop.run_until_complete(
            self._batch_async([self._prompt_async(prompt) for prompt in prompts])
        )

        return cast(List[TLMResponse], tlm_responses)

    def _batch_get_trustworthiness_score(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
    ) -> List[float]:
        """Run batch of TLM get confidence score.

        Args:
            prompts (Sequence[str]): list of prompts to run get confidence score for
            responses (Sequence[str]): list of responses to run get confidence score for

        Returns:
            List[float]: TLM confidence score for each prompt (in supplied order)
        """
        tlm_responses = self._event_loop.run_until_complete(
            self._batch_async(
                [
                    self._get_trustworthiness_score_async(prompt, response)
                    for prompt, response in zip(prompts, responses)
                ]
            )
        )

        return cast(List[float], tlm_responses)

    async def _batch_async(
        self,
        tlm_coroutines: List[
            Union[Coroutine[None, None, TLMResponse], Coroutine[None, None, float]]
        ],
    ) -> Union[List[TLMResponse], List[float]]:
        tlm_query_tasks = [asyncio.create_task(tlm_coro) for tlm_coro in tlm_coroutines]

        if self._verbose:
            return await asyncio.wait_for(
                tqdm_asyncio.gather(
                    *tlm_query_tasks,
                    total=len(tlm_query_tasks),
                    desc="Querying TLM...",
                    bar_format="{desc} {percentage:3.0f}%|{bar}|",
                ),
                timeout=self._timeout,
            )
        return await asyncio.wait_for(asyncio.gather(*tlm_query_tasks), timeout=self._timeout)  # type: ignore[arg-type]

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
            return self._event_loop.run_until_complete(
                asyncio.wait_for(self._prompt_async(prompt), timeout=self._timeout)
            )

        return self._batch_prompt(prompt)

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
                return await asyncio.wait_for(
                    self._prompt_async(prompt, session), timeout=self._timeout
                )

            return cast(
                List[TLMResponse],
                await self._batch_async([self._prompt_async(p, session) for p in prompt]),
            )

    async def _prompt_async(
        self,
        prompt: str,
        client_session: Optional[aiohttp.ClientSession] = None,
    ) -> TLMResponse:
        """
        Private asynchronous method to get response and trustworthiness score from TLM.

        Args:
            prompt (str): prompt for the TLM
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

        Args:
            prompt (str | Sequence[str]): prompt (or list of multiple prompts) for the TLM
            response (str | Sequence[str]): response (or list of multiple responses) for the TLM to evaluate
        Returns:
            float (or list of floats if multiple prompt-responses were provided) corresponding to the TLM's trustworthiness score.
                    The score quantifies how confident TLM is that the given response is good for the given prompt.
        """
        validate_tlm_prompt_response(prompt, response)

        if isinstance(prompt, str) and isinstance(response, str):
            return self._event_loop.run_until_complete(
                asyncio.wait_for(
                    self._get_trustworthiness_score_async(prompt, response), timeout=self._timeout
                )
            )

        return self._batch_get_trustworthiness_score(prompt, response)

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
                return await asyncio.wait_for(
                    self._get_trustworthiness_score_async(prompt, response, session),
                    timeout=self._timeout,
                )

            return cast(
                List[float],
                await self._batch_async(
                    [
                        self._get_trustworthiness_score_async(p, r, session)
                        for p, r in zip(prompt, response)
                    ]
                ),
            )

    async def _get_trustworthiness_score_async(
        self,
        prompt: str,
        response: str,
        client_session: Optional[aiohttp.ClientSession] = None,
    ) -> float:
        """Private asynchronous method to get trustworthiness score for prompt-response pair.

        Args:
            prompt: prompt for the TLM
            response: response for the TLM to evaluate
            client_session: async HTTP session to use for TLM query. Defaults to None.
        Returns:
            float corresponding to the TLM's trustworthiness score

        """
        if self._quality_preset == "base":
            raise ValueError(
                "Cannot get confidence score with `base` quality_preset -- choose a higher preset."
            )

        async with self._query_semaphore:
            return cast(
                float,
                (
                    await api.tlm_get_confidence_score(
                        self._api_key,
                        prompt,
                        response,
                        self._quality_preset,
                        self._options,
                        client_session,
                        retries=_TLM_MAX_RETRIES,
                    )
                )["confidence_score"],
            )


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
