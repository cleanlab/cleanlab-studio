"""
Cleanlab TLM is a Large Language Model that gives more reliable answers and quantifies its uncertainty in these answers
"""

from __future__ import annotations

import asyncio
import sys
from typing import Coroutine, List, Literal, Optional, Union, cast
from tqdm.asyncio import tqdm_asyncio

import aiohttp
from typing_extensions import NotRequired, TypedDict  # for Python <3.11 with (Not)Required

from cleanlab_studio.internal.api import api
from cleanlab_studio.internal.types import JSONDict

valid_quality_presets = ["best", "high", "medium", "low", "base"]
QualityPreset = Literal["best", "high", "medium", "low", "base"]

valid_tlm_models = ["gpt-3.5-turbo-16k", "gpt-4"]
TLMModel = Literal["gpt-3.5-turbo-16k", "gpt-4"]

DEFAULT_VERBOSITY: bool = True
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
    """Trustworthy language model options. The TLM quality-preset determines many of these settings automatically, but
    specifying other values here will over-ride the setting from the quality-preset.

    Parameters
    ----------
    max_tokens: int, default = 512
       The maximum number of tokens to generate in the TLM response.

    max_timeout: int, optional
       The maximum timeout to query from TLM in seconds. If a max_timeout is not specified, then timeout is calculated based on number of tokens.

    num_candidate_responses: int, default = 1
       This controls how many candidate responses are internally generated.
       TLM scores the confidence of each candidate response, and then returns the most confident one.
       A higher value here can produce better (more accurate) responses from the TLM, but at higher costs/runtimes.

    num_consistency_samples: int, default = 5
       This controls how many samples are internally generated to evaluate the LLM-response-consistency.
       This is a big part of the returned confidence_score, in particular for ensuring lower scores for strange input prompts or those that are too open-ended to receive a well-defined 'good' response.
       Higher values here produce better (more reliable) TLM confidence scores, but at higher costs/runtimes.

    use_self_reflection: bool, default = True
       This controls whether self-reflection is used to have the LLM reflect upon the response it is generating and explicitly self-evaluate whether it seems good or not.
       This is a big part of the confidence score, in particular for ensure low scores for responses that are obviously incorrect/bad for a standard prompt that LLMs should be able to handle.
       Setting this to False disables the use of self-reflection and may produce worse TLM confidence scores, but can reduce costs/runtimes.
    """

    max_tokens: NotRequired[int]
    max_timeout: NotRequired[int]
    num_candidate_responses: NotRequired[int]
    num_consistency_samples: NotRequired[int]
    use_self_reflection: NotRequired[bool]


class TLM:
    """TLM interface class."""

    def __init__(
        self,
        api_key: str,
        quality_preset: QualityPreset,
        model: TLMModel,
        verbosity: bool,
        max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_TLM_REQUESTS,
    ) -> None:
        """Initializes TLM interface.

        Parameters
        ----------
            api_key: str
                API key used to authenticate TLM client
            quality_preset: QualityPreset, default = "low"
                Quality preset to use for TLM queries
            max_concurrent_requests: int, default = 16
                Maximum number of concurrent requests when issuing batch queries. Default is 16.
            model: TLMModel, default = "gpt-3.5-turbo-16k"
                ID of the model to use. Other options: "gpt-4"
            verbosity: bool, default = True
                Verbosity level for TLM queries. Default is True which will print progress bars for TLM queries. For silent TLM progress, set to False.
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

        if model not in valid_tlm_models:
            raise ValueError(f"Invalid model {model} -- must be one of {valid_tlm_models}")
        self._model = model

        if is_notebook():
            import nest_asyncio

            nest_asyncio.apply()

        self._event_loop = asyncio.get_event_loop()
        self._query_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._verbosity = verbosity

    def batch_prompt(
        self,
        prompts: List[str],
        options: Union[None, TLMOptions, List[Union[TLMOptions, None]]] = None,
        timeout: Optional[float] = None,
        retries: int = 0,
    ) -> List[TLMResponse]:
        """Run batch of TLM prompts.

        Args:
            prompts (List[str]): list of prompts to run
            options (None | TLMOptions | List[TLMOptions  |  None], optional): list of options (or instance of options) to pass to prompt method. Defaults to None.
            timeout (Optional[float], optional): timeout (in seconds) to run all prompts. Defaults to None.
            retries (int): number of retries to attempt for each individual prompt. Defaults to 0.

        Returns:
            List[TLMResponse]: TLM responses for each prompt (in supplied order)
        """
        if not isinstance(options, list):
            options = [options for _ in prompts]

        assert len(prompts) == len(options), "Length of prompts and options must match."

        tlm_responses = self._event_loop.run_until_complete(
            self._batch_async(
                [
                    self.prompt_async(
                        prompt,
                        option_dict,
                        retries=retries,
                    )
                    for prompt, option_dict in zip(prompts, options)
                ],
                timeout=timeout,
            )
        )

        return cast(List[TLMResponse], tlm_responses)

    def batch_get_confidence_score(
        self,
        prompts: List[str],
        responses: List[str],
        options: Union[None, TLMOptions, List[Union[TLMOptions, None]]] = None,
        timeout: Optional[float] = None,
        retries: int = 0,
    ) -> List[float]:
        """Run batch of TLM get confidence score.

        Args:
            prompts (List[str]): list of prompts to run get confidence score for
            responses (List[str]): list of responses to run get confidence score for
            options (None | TLMOptions | List[TLMOptions  |  None], optional): list of options (or instance of options) to pass to get confidence score method. Defaults to None.
            timeout (Optional[float], optional): timeout (in seconds) to run all prompts. Defaults to None.
            retries (int): number of retries to attempt for each individual prompt. Defaults to 0.

        Returns:
            List[float]: TLM confidence score for each prompt (in supplied order)
        """
        if not isinstance(options, list):
            options = [options for _ in prompts]

        assert len(prompts) == len(responses), "Length of prompts and responses must match."
        assert len(prompts) == len(options), "Length of prompts and options must match."

        tlm_responses = self._event_loop.run_until_complete(
            self._batch_async(
                [
                    self.get_confidence_score_async(
                        prompt,
                        response,
                        option_dict,
                        retries=retries,
                    )
                    for prompt, response, option_dict in zip(prompts, responses, options)
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

        if self._verbosity:
            return await asyncio.wait_for(
                tqdm_asyncio.gather(
                    *tlm_query_tasks,
                    total=len(tlm_query_tasks),
                    desc="Querying TLM...",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}|",
                ),
                timeout=timeout,
            )
        else:
            return await asyncio.wait_for(
                asyncio.gather(*tlm_query_tasks),
                timeout=timeout,
            )

    def prompt(self, prompt: str, options: Optional[TLMOptions] = None) -> TLMResponse:
        """
        Get response and confidence from TLM.

        Args:
            prompt (str): prompt for the TLM
            options (Optional[TLMOptions]): options to parameterize TLM with. Defaults to None.
        Returns:
            TLMResponse: [TLMResponse](#class-tlmresponse) object containing the response and confidence score
        """
        return self._event_loop.run_until_complete(
            self.prompt_async(
                prompt,
                options,
            )
        )

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
        self, prompt: str, response: str, options: Optional[TLMOptions] = None
    ) -> float:
        """Gets confidence score for prompt-response pair.

        Args:
            prompt: prompt for the TLM
            response: response for the TLM to evaluate
            options (Optional[TLMOptions]): options to parameterize TLM with. Defaults to None.
        Returns:
            float corresponding to the TLM's confidence score
        """
        return self._event_loop.run_until_complete(
            self.get_confidence_score_async(
                prompt,
                response,
                options,
            )
        )

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
