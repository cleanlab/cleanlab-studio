"""
Cleanlab TLM is a Large Language Model that gives more reliable answers and quantifies its uncertainty in these answers
"""
from __future__ import annotations

import asyncio
import sys
from typing import Coroutine, Collection, List, Literal, Optional, Union, cast

import aiohttp
from typing_extensions import NotRequired, TypedDict  # for Python <3.11 with (Not)Required

from cleanlab_studio.internal.api import api
from cleanlab_studio.internal.api.api_helper import is_collection
from cleanlab_studio.internal.types import JSONDict

valid_quality_presets = ["best", "high", "medium", "low", "base"]
QualityPreset = Literal["best", "high", "medium", "low", "base"]

DEFAULT_MAX_CONCURRENT_TLM_REQUESTS: int = 16
MAX_CONCURRENT_TLM_REQUESTS_LIMIT: int = 128


class TLMResponse(TypedDict):
    """Trustworthy Language Model response.

    Attributes:
        response (str): text response from language model
        confidence_score (float): score corresponding to confidence that the response is correct
    """

    response: str
    confidence_score: float


class TLMOptions(TypedDict):
    """Trustworthy language model options.

    Attributes:
        max_tokens (int): the maximum number of tokens to generate in the TLM response
        model (str): ID of the model to use. Default: "gpt-3.5-turbo-16k". Other options: "gpt-4"
    """

    max_tokens: int
    model: NotRequired[str]


class TLM:
    """TLM interface class."""

    def __init__(
        self,
        api_key: str,
        quality_preset: QualityPreset,
        max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_TLM_REQUESTS,
    ) -> None:
        """Initializes TLM interface.

        Args:
            api_key (str): API key used to authenticate TLM client
            quality_preset (QualityPreset): quality preset to use for TLM queries
            max_concurrent_requests (int): maximum number of concurrent requests when issuing batch queries. Default is 16.
        """
        self._api_key = api_key

        assert (
            max_concurrent_requests < MAX_CONCURRENT_TLM_REQUESTS_LIMIT
        ), f"max_concurrent_requests must be less than {MAX_CONCURRENT_TLM_REQUESTS_LIMIT}"

        if quality_preset not in valid_quality_presets:
            raise ValueError(
                f"Invalid quality preset {quality_preset} -- must be one of {valid_quality_presets}"
            )

        self._quality_preset = quality_preset

        if is_notebook():
            import nest_asyncio

            nest_asyncio.apply()

        self._event_loop = asyncio.get_event_loop()
        self._query_semaphore = asyncio.Semaphore(max_concurrent_requests)

    def _batch_prompt(
        self,
        prompts: Collection[str],
        options: Union[None, TLMOptions, Collection[Union[TLMOptions, None]]] = None,
        timeout: Optional[float] = None,
        retries: int = 1,
    ) -> List[TLMResponse]:
        """Run batch of TLM prompts.

        Args:
            prompts (Collection[str]): list or other iterable of prompts to run
            options (None | TLMOptions | List[TLMOptions  |  None], optional): collection of options (or instance of options) to pass to prompt method. Defaults to None.
            timeout (Optional[float], optional): timeout (in seconds) to run all prompts. Defaults to None.
            retries (int): number of retries to attempt for each individual prompt in case of error. Defaults to 1.

        Returns:
            List[TLMResponse]: TLM responses for each prompt (in supplied order)
        """
        if is_collection(options):
            options_collection = cast(Collection[Union[TLMOptions, None]], options)
        else:
            options = cast(Union[None, TLMOptions], options)
            options_collection = [options for _ in prompts]

        assert len(prompts) == len(options_collection), "Length of prompts and options must match."

        tlm_responses = self._event_loop.run_until_complete(
            self._batch_async(
                [
                    self.prompt_async(
                        prompt,
                        option_dict,
                        retries=retries,
                    )
                    for prompt, option_dict in zip(prompts, options_collection)
                ],
                timeout=timeout,
            )
        )

        return cast(List[TLMResponse], tlm_responses)

    def _batch_get_confidence_score(
        self,
        prompts: Collection[str],
        responses: Collection[str],
        options: Union[None, TLMOptions, Collection[Union[TLMOptions, None]]] = None,
        timeout: Optional[float] = None,
        retries: int = 1,
    ) -> List[float]:
        """Run batch of TLM get confidence score.

        Args:
            prompts (Collection[str]): list or other iterable of prompts to run get confidence score for
            responses (Collection[str]): list or other iterable of responses to run get confidence score for
            options (None | TLMOptions | List[TLMOptions  |  None], optional): list of options (or instance of options) to pass to get confidence score method. Defaults to None.
            timeout (Optional[float], optional): timeout (in seconds) to run all prompts. Defaults to None.
            retries (int): number of retries to attempt for each individual prompt in case of error. Defaults to 1.

        Returns:
            List[float]: TLM confidence score for each prompt (in supplied order)
        """
        if is_collection(options):
            options_collection = cast(Collection[Union[TLMOptions, None]], options)
        else:
            options = cast(Union[None, TLMOptions], options)
            options_collection = [options for _ in prompts]

        assert len(prompts) == len(responses), "Length of prompts and responses must match."
        assert len(prompts) == len(options_collection), "Length of prompts and options must match."

        tlm_responses = self._event_loop.run_until_complete(
            self._batch_async(
                [
                    self.get_confidence_score_async(
                        prompt,
                        response,
                        option_dict,
                        retries=retries,
                    )
                    for prompt, response, option_dict in zip(prompts, responses, options_collection)
                ],
                timeout=timeout,
            )
        )

        return cast(List[float], tlm_responses)

    async def _batch_async(
        self,
        tlm_coroutines: List[
            Union[Coroutine[None, None, TLMResponse], Coroutine[None, None, float]]
        ],
        timeout: Optional[float],
    ) -> Union[List[TLMResponse], List[float]]:
        tlm_query_tasks = [asyncio.create_task(tlm_coro) for tlm_coro in tlm_coroutines]

        return await asyncio.wait_for(asyncio.gather(*tlm_query_tasks), timeout=timeout)  # type: ignore[arg-type]

    def prompt(
        self,
        prompt: Union[str, Collection[str]],
        options: Union[None, TLMOptions, Collection[Union[TLMOptions, None]]] = None,
        timeout: Optional[float] = None,
        retries: int = 1,
    ) -> Union[TLMResponse, List[TLMResponse]]:
        """
        Get response and confidence from TLM.

        Args:
            prompt (str | Collection[str]): prompt (or list/iterable of multiple prompts) for the TLM
            options (None | TLMOptions | Collection[TLMOptions |  None], optional): collection of options (or instance of options) to pass to prompt method. Defaults to None.
            timeout (Optional[float], optional): timeout (in seconds) to run all prompts. Defaults to None.
                If the timeout is hit, this method will throw a `TimeoutError`.
                Larger values give TLM a higher chance to return outputs for all of your prompts.
                Smaller values ensure this method does not take too long.
            retries (int): number of retries to attempt for each individual prompt in case of internal error. Defaults to 1.
                Larger values give TLM a higher chance of returning outputs for all of your prompts,
                but this method will also take longer to alert you in cases of an unrecoverable error.
                Set to 0 to never attempt any retries.
        Returns:
            TLMResponse | List[TLMResponse]: [TLMResponse](#class-tlmresponse) object containing the response and confidence score.
                    If multiple prompts were provided in a list, then a list of such objects is returned, one for each prompt.
        """
        if is_collection(prompt):
            return self._batch_prompt(
                prompt,
                options,
                timeout=timeout,
                retries=retries,
            )

        elif isinstance(prompt, str):
            if not (options is None or isinstance(options, dict)):
                raise ValueError("options must be a single TLMOptions object for single prompt.")

            return self._event_loop.run_until_complete(
                self.prompt_async(
                    prompt,
                    cast(Union[None, TLMOptions], options),
                    retries=retries,
                )
            )

        else:
            raise ValueError("prompt must be a string or list/iterable of strings.")

    async def prompt_async(
        self,
        prompt: str,
        options: Optional[TLMOptions] = None,
        client_session: Optional[aiohttp.ClientSession] = None,
        retries: int = 0,
    ) -> TLMResponse:
        """
        (Asynchronously) Get response and confidence from TLM.

        Args:
            prompt (str): prompt for the TLM
            options (Optional[TLMOptions]): options to parameterize TLM with. Defaults to None.
            client_session (Optional[aiohttp.ClientSession]): async HTTP session to use for TLM query. Defaults to None.
            retries (int): number of retries for TLM query. Defaults to 0.
        Returns:
            TLMResponse: [TLMResponse](#class-tlmresponse) object containing the response and confidence score
        """
        async with self._query_semaphore:
            tlm_response = await api.tlm_prompt(
                self._api_key,
                prompt,
                self._quality_preset,
                cast(JSONDict, options),
                client_session,
                retries=retries,
            )

        return {
            "response": tlm_response["response"],
            "confidence_score": tlm_response["confidence_score"],
        }

    def get_confidence_score(
        self,
        prompt: Union[str, Collection[str]],
        response: Union[str, Collection[str]],
        options: Union[None, TLMOptions, Collection[Union[TLMOptions, None]]] = None,
        timeout: Optional[float] = None,
        retries: int = 1,
    ) -> Union[float, List[float]]:
        """Gets confidence score for prompt-response pair(s).

        Args:
            prompt (str | Collection[str]): prompt (or list/iterable of multiple prompts) for the TLM
            response (str | Collection[str]): response (or list/iterable of multiple responses) for the TLM to evaluate
            options (None | TLMOptions | List[TLMOptions  |  None], optional): list of options (or instance of options) to pass to get confidence score method. Defaults to None.
            timeout (Optional[float], optional): maximum allowed time (in seconds) to run all prompts and evaluate all responses. Defaults to None.
                If the timeout is hit, this method will throw a `TimeoutError`.
                Larger values give TLM a higher chance to return outputs for all of your prompts + responses.
                Smaller values ensure this method does not take too long.
            retries (int): number of retries to attempt for each individual prompt in case of internal error. Defaults to 1.
                Larger values give TLM a higher chance of returning outputs for all of your prompts,
                but this method will also take longer to alert you in cases of an unrecoverable error.
                Set to 0 to never attempt any retries.
        Returns:
            float (or list of floats if multiple prompt-responses were provided) corresponding to the TLM's confidence score.
                    The score quantifies how confident TLM is that the given response is good for the given prompt.
        """
        if is_collection(prompt):
            if not is_collection(response):
                raise ValueError(
                    "responses must be a list or iterable of strings when prompt is a list or iterable."
                )

            return self._batch_get_confidence_score(
                prompt,
                response,
                options,
                timeout=timeout,
                retries=retries,
            )

        elif isinstance(prompt, str):
            if not (options is None or isinstance(options, dict)):
                raise ValueError("options must be a single TLMOptions object for single prompt.")

            if not isinstance(response, str):
                raise ValueError("responses must be a single string for single prompt.")

            return self._event_loop.run_until_complete(
                self.get_confidence_score_async(
                    prompt,
                    response,
                    cast(Union[None, TLMOptions], options),
                    retries=retries,
                )
            )

        else:
            raise ValueError("prompt must be a string or list/iterable of strings.")

    async def get_confidence_score_async(
        self,
        prompt: str,
        response: str,
        options: Optional[TLMOptions] = None,
        client_session: Optional[aiohttp.ClientSession] = None,
        retries: int = 0,
    ) -> float:
        """(Asynchronously) gets confidence score for prompt-response pair.

        Args:
            prompt: prompt for the TLM
            response: response for the TLM  to evaluate
            options (Optional[TLMOptions]): options to parameterize TLM with. Defaults to None.
            client_session (Optional[aiohttp.ClientSession]): async HTTP session to use for TLM query. Defaults to None.
            retries (int): number of retries for TLM query. Defaults to 0.
        Returns:
            float corresponding to the TLM's confidence score

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
                        cast(JSONDict, options),
                        client_session,
                        retries=retries,
                    )
                )["confidence_score"],
            )


def is_notebook() -> bool:
    """Returns True if running in a notebook, False otherwise."""
    try:
        get_ipython = sys.modules["IPython"].get_ipython
        if "IPKernelApp" in get_ipython().config:
            return True

        return False
    except:
        return False
