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
from cleanlab_studio.internal.tlm.concurrency import TlmRateHandler
from cleanlab_studio.internal.tlm.validation import (
    validate_tlm_prompt,
    validate_tlm_try_prompt,
    validate_tlm_prompt_response,
    validate_try_tlm_prompt_response,
    validate_tlm_options,
)
from cleanlab_studio.internal.types import TLMQualityPreset
from cleanlab_studio.errors import ValidationError
from cleanlab_studio.internal.constants import (
    _VALID_TLM_QUALITY_PRESETS,
    _TLM_MAX_RETRIES,
)


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
            api_key (str): API key used to authenticate TLM client.
            Cleanlab Studio API keys can be obtained on the [Account](app.cleanlab.ai/account?tab=General) page.

            quality_preset (TLMQualityPreset): quality preset to use for TLM queries, which will determine the quality of the output responses and trustworthiness scores.
            Supported presets include "best", "high", "medium", "low", "base".
            The "best" and "high" presets will improve the LLM responses themselves, with "best" also returning the most reliable trustworthiness scores.
            The "medium" and "low" presets will return standard LLM responses along with associated confidence scores,
            with "medium" producing more reliable trustworthiness scores than low.
            The "base" preset will not return any confidence score, just a standard LLM output response, this option is similar to using your favorite LLM API.

            options (TLMOptions, optional): a typed dictionary of options to pass to prompt method, defaults to None.
            Options that can be passed in include "model", "max_tokens", "num_candidate_responses", "num_consistency_samples", "use_self_reflection".
            For more details about the options, see the documentation for [TLMOptions](#class-tlmoptions).

            timeout (float, optional): timeout (in seconds) to run all prompts, defaults to None which does not apply a timeout.

            verbose (bool, optional): verbosity level for TLM queries. For silent TLM progress, set to False.
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
            raise ValidationError("verbose must be a boolean value")

        is_notebook_flag = is_notebook()

        self._quality_preset = quality_preset
        self._options = options
        self._timeout = timeout if timeout is not None and timeout > 0 else None
        self._verbose = verbose if verbose is not None else is_notebook_flag

        if is_notebook_flag:
            import nest_asyncio

            nest_asyncio.apply()

        self._event_loop = asyncio.get_event_loop()
        self._rate_handler = TlmRateHandler()

    async def _batch_prompt(
        self,
        prompts: Sequence[str],
        capture_exceptions: bool = False,
    ) -> Union[BatchPromptResponse, TryBatchPromptResponse]:
        """Run batch of TLM prompts. The list returned will have the same length as the input list.

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
                    prompt,
                    timeout=per_query_timeout,
                    capture_exceptions=capture_exceptions,
                    batch_index=batch_index,
                )
                for batch_index, prompt in enumerate(prompts)
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
            Union[BatchGetTrustworthinessScoreResponse, TryBatchGetTrustworthinessScoreResponse]: TLM confidence score for each prompt (in supplied order)
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
                    batch_index=batch_index,
                )
                for batch_index, (prompt, response) in enumerate(zip(prompts, responses))
            ],
            per_batch_timeout,
        )

        if capture_exceptions:
            return cast(TryBatchGetTrustworthinessScoreResponse, tlm_responses)

        return cast(BatchGetTrustworthinessScoreResponse, tlm_responses)

    async def _batch_async(
        self,
        tlm_coroutines: Sequence[Coroutine[None, None, Union[TLMResponse, float, None]]],
        batch_timeout: Optional[float] = None,
    ) -> Sequence[Union[TLMResponse, float, None]]:
        """Runs batch of TLM queries.

        Args:
            tlm_coroutines (List[Coroutine[None, None, Union[TLMResponse, float, None]]]): list of query coroutines to run, returning TLM responses or confidence scores (or None if capture_exceptions is True)
            batch_timeout (Optional[float], optional): timeout (in seconds) to run all queries, defaults to None (no timeout)

        Returns:
            Sequence[Union[TLMResponse, float, None]]: list of coroutine results, with preserved order
        """
        tlm_query_tasks = [asyncio.create_task(tlm_coro) for tlm_coro in tlm_coroutines]

        if self._verbose:
            gather_task = tqdm_asyncio.gather(
                *tlm_query_tasks,
                total=len(tlm_query_tasks),
                desc="Querying TLM...",
                bar_format="{desc} {percentage:3.0f}%|{bar}|",
            )
        else:
            gather_task = asyncio.gather(*tlm_query_tasks)

        wait_task = asyncio.wait_for(gather_task, timeout=batch_timeout)
        try:
            return cast(
                Sequence[Union[TLMResponse, float, None]],
                await wait_task,
            )
        except Exception:
            # if exception occurs while awaiting batch results, cancel remaining tasks
            for query_task in tlm_query_tasks:
                query_task.cancel()

            # await remaining tasks to ensure they are cancelled
            await asyncio.gather(*tlm_query_tasks, return_exceptions=True)

            raise

    def prompt(
        self,
        prompt: Union[str, Sequence[str]],
        /,
    ) -> Union[TLMResponse, BatchPromptResponse]:
        """
        Get response and trustworthiness score for any text input from TLM.

        Args:
            prompt (str | Sequence[str]): prompt (or list of multiple prompts) for the TLM
        Returns:
            TLMResponse | BatchPromptResponse: [TLMResponse](#class-tlmresponse) object containing the response and trustworthiness score.
                If multiple prompts were provided in a list, then a list of such objects is returned, one for each prompt.
                This method will raise an exception if any errors occur or if you hit a timeout (given a timeout is specified),
                and is suitable if strict error handling and immediate notification of any exceptions/timeouts is preferred.
                However, you could lose any partial results if an exception is raised.
                If saving partial results is important to you, check out the [try_prompt](#method-try_prompt) method instead.
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
    ) -> TryBatchPromptResponse:
        """
        Get response and trustworthiness score for any text input from TLM,
        and handles any failures (errors of timeouts) by returning None in place of the failures.

        The list returned will have the same length as the input list, if there are any
        failures (errors or timeout) processing some inputs, the list will contain None in place of the response.

        If there are any failures (errors or timeouts) processing some inputs, the list returned will have
        the same length as the input list. In case of failure, the list will contain None in place of the response.

        Args:
            prompt (Sequence[str]): list of multiple prompts for the TLM
        Returns:
            TryBatchPromptResponse: list of [TLMResponse](#class-tlmresponse) objects containing the response and trustworthiness score.
                The returned list will always have the same length as the input list.
                In case of failure on any prompt (due to timeouts or other erros),
                the return list will contain None in place of the TLM response.
                This method is suitable if you prioritize obtaining results for as many inputs as possible,
                however you might miss out on certain error messages.
                If you would prefer to be notified immediately about any errors or timeouts that might occur,
                check out the [prompt](#method-prompt) method instead.
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
    ) -> Union[TLMResponse, BatchPromptResponse]:
        """
        Asynchronously get response and trustworthiness score for any text input from TLM.
        This method is similar to the [prompt](#method-prompt) method but operates asynchronously.

        Args:
            prompt (str | Sequence[str]): prompt (or list of multiple prompts) for the TLM
        Returns:
            TLMResponse | BatchPromptResponse: [TLMResponse](#class-tlmresponse) object containing the response and trustworthiness score.
                If multiple prompts were provided in a list, then a list of such objects is returned, one for each prompt.
                This method will raise an exception if any errors occur or if you hit a timeout (given a timeout is specified).
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
        batch_index: Optional[int] = None,
    ) -> Optional[TLMResponse]:
        """
        Private asynchronous method to get response and trustworthiness score from TLM.

        Args:
            prompt (str): prompt for the TLM
            client_session (aiohttp.ClientSession, optional): async HTTP session to use for TLM query. Defaults to None (creates a new session).
            timeout: timeout (in seconds) to run the prompt, defaults to None (no timeout)
            capture_exceptions: if should return None in place of the response for any errors
            batch_index: index of the prompt in the batch, used for error messages
        Returns:
            TLMResponse: [TLMResponse](#class-tlmresponse) object containing the response and trustworthiness score.
        """

        try:
            tlm_response = await asyncio.wait_for(
                api.tlm_prompt(
                    self._api_key,
                    prompt,
                    self._quality_preset,
                    self._options,
                    self._rate_handler,
                    client_session,
                    batch_index=batch_index,
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
    ) -> Union[float, BatchGetTrustworthinessScoreResponse]:
        """Gets trustworthiness score for prompt-response pairs.

        Args:
            prompt (str | Sequence[str]): prompt (or list of prompts) for the TLM to evaluate
            response (str | Sequence[str]): response (or list of responses) corresponding to the input prompts
        Returns:
            float | List[float]: float or list of floats (if multiple prompt-responses were provided) corresponding
                to the TLM's trustworthiness score.
                The score quantifies how confident TLM is that the given response is good for the given prompt.
                This method will raise an exception if any errors occur or if you hit a timeout (given a timeout is specified),
                and is suitable if strict error handling and immediate notification of any exceptions/timeouts is preferred.
                However, you could lose any partial results if an exception is raised.
                If saving partial results is important to you, check out the [try_get_trustworthiness_score](#method-try_get_trustworthiness_score) method instead.
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
            BatchGetTrustworthinessScoreResponse,
            self._event_loop.run_until_complete(
                self._batch_get_trustworthiness_score(prompt, response, capture_exceptions=False)
            ),
        )

    def try_get_trustworthiness_score(
        self,
        prompt: Sequence[str],
        response: Sequence[str],
    ) -> TryBatchGetTrustworthinessScoreResponse:
        """Gets trustworthiness score for prompt-response pairs.
        The list returned will have the same length as the input list, if there are any
        failures (errors or timeout) processing some inputs, the list will contain None
        in place of the response.

        Args:
            prompt (Sequence[str]): list of prompts for the TLM to evaluate
            response (Sequence[str]): list of responses corresponding to the input prompts
        Returns:
            List[float]: list of floats corresponding to the TLM's trustworthiness score.
                The score quantifies how confident TLM is that the given response is good for the given prompt.
                The returned list will always have the same length as the input list.
                In case of failure on any prompt-response pair (due to timeouts or other erros),
                the return list will contain None in place of the trustworthiness score.
                This method is suitable if you prioritize obtaining results for as many inputs as possible,
                however you might miss out on certain error messages.
                If you would prefer to be notified immediately about any errors or timeouts that might occur,
                check out the [get_trustworthiness_score](#method-get_trustworthiness_score) method instead.
        """
        validate_try_tlm_prompt_response(prompt, response)

        return cast(
            TryBatchGetTrustworthinessScoreResponse,
            self._event_loop.run_until_complete(
                self._batch_get_trustworthiness_score(prompt, response, capture_exceptions=True)
            ),
        )

    async def get_trustworthiness_score_async(
        self,
        prompt: Union[str, Sequence[str]],
        response: Union[str, Sequence[str]],
    ) -> Union[float, BatchGetTrustworthinessScoreResponse]:
        """Asynchronously gets trustworthiness score for prompt-response pairs.
        This method is similar to the [get_trustworthiness_score](#method-get_trustworthiness_score) method but operates asynchronously.

        Args:
            prompt (str | Sequence[str]): prompt (or list of prompts) for the TLM to evaluate
            response (str | Sequence[str]): response (or list of responses) corresponding to the input prompts
        Returns:
            float | List[float]: float or list of floats (if multiple prompt-responses were provided) corresponding
                to the TLM's trustworthiness score.
                The score quantifies how confident TLM is that the given response is good for the given prompt.
                This method will raise an exception if any errors occur or if you hit a timeout (given a timeout is specified).
        """
        validate_tlm_prompt_response(prompt, response)

        async with aiohttp.ClientSession() as session:
            if isinstance(prompt, str) and isinstance(response, str):
                trustworthiness_score = await self._get_trustworthiness_score_async(
                    prompt, response, session, timeout=self._timeout, capture_exceptions=False
                )
                return cast(float, trustworthiness_score)

            return cast(
                BatchGetTrustworthinessScoreResponse,
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
        batch_index: Optional[int] = None,
    ) -> Optional[float]:
        """Private asynchronous method to get trustworthiness score for prompt-response pairs.

        Args:
            prompt: prompt for the TLM to evaluate
            response: response corresponding to the input prompt
            client_session: async HTTP session to use for TLM query. Defaults to None.
            timeout: timeout (in seconds) to run the prompt, defaults to None (no timeout)
            capture_exceptions: if should return None in place of the response for any errors
            batch_index: index of the prompt in the batch, used for error messages
        Returns:
            float corresponding to the TLM's trustworthiness score

        """
        if self._quality_preset == "base":
            raise ValueError(
                "Cannot get confidence score with `base` quality_preset -- choose a higher preset."
            )

        try:
            tlm_response = await asyncio.wait_for(
                api.tlm_get_confidence_score(
                    self._api_key,
                    prompt,
                    response,
                    self._quality_preset,
                    self._options,
                    self._rate_handler,
                    client_session,
                    batch_index=batch_index,
                    retries=_TLM_MAX_RETRIES,
                ),
                timeout=timeout,
            )

            return cast(float, tlm_response["confidence_score"])

        except Exception as e:
            if capture_exceptions:
                return None
            raise e


class TLMResponse(TypedDict):
    """A typed dict containing the response and trustworthiness score from the Trustworthy Language Model.

    Attributes:
        response (str): text response from the Trustworthy Language Model
        trustworthiness_score (float, optional): score between 0-1 corresponding to the trustworthiness of the response.
        A higher score indicates a higher confidence that the response is correct/trustworthy.
    """

    response: str
    trustworthiness_score: Optional[float]


class TLMOptions(TypedDict):
    """Typed dict containing options for the Trustworthy Language Model.
    Many of these arguments are automatically determined by the quality preset selected
    (see the arguments in the TLM [initialization method](#method-__init__) to learn more about the various quality presets),
    but specifying custom values here will override any default values from the quality preset.

    Args:
        model (str, default = "gpt-3.5-turbo-16k"): underlying LLM to use (better models will yield better results).
        Models currently supported include "gpt-3.5-turbo-16k", "gpt-4".

        max_tokens (int, default = 512): the maximum number of tokens to generate in the TLM response.
        The minimum value for this parameter is 64, and the maximum is 512.

        num_candidate_responses (int, default = 1): this controls how many candidate responses are internally generated.
        TLM scores the trustworthiness of each candidate response, and then returns the most trustworthy one.
        Higher values here can produce better (more accurate) responses from the TLM, but at higher costs/runtimes.
        The minimum value for this parameter is 1, and the maximum is 20.

        num_consistency_samples (int, default = 5): this controls how many samples are internally generated to evaluate the LLM-response-consistency.
        This is a big part of the returned trustworthiness_score, in particular to evaluate strange input prompts or prompts that are too open-ended
        to receive a clearly defined 'good' response.
        Higher values here produce better (more reliable) TLM confidence scores, but at higher costs/runtimes.
        The minimum value for this parameter is 0, and the maximum is 20.

        use_self_reflection (bool, default = `True`): this controls whether self-reflection is used to have the LLM reflect upon the response it is
        generating and explicitly self-evaluate the accuracy of that response.
        This is a big part of the trustworthiness score, in particular for evaluating responses that are obviously incorrect/bad for a
        standard prompt (with well-defined answers) that LLMs should be able to handle.
        Setting this to False disables the use of self-reflection and may produce worse TLM trustworthiness scores, but will reduce costs/runtimes.
    """

    model: NotRequired[str]
    max_tokens: NotRequired[int]
    num_candidate_responses: NotRequired[int]
    num_consistency_samples: NotRequired[int]
    use_self_reflection: NotRequired[bool]


BatchPromptResponse = List[TLMResponse]
TryBatchPromptResponse = List[Optional[TLMResponse]]
BatchGetTrustworthinessScoreResponse = List[float]
TryBatchGetTrustworthinessScoreResponse = List[Optional[float]]


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
