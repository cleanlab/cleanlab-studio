"""
Cleanlab TLM is a Large Language Model that gives more reliable answers and quantifies its uncertainty in these answers
"""
import asyncio
from typing import cast, Literal, Optional, TypedDict

import aiohttp

from cleanlab_studio.internal.api import api
from cleanlab_studio.internal.types import JSONDict


valid_quality_presets = ["best", "high", "medium", "low", "base"]
QualityPreset = Literal["best", "high", "medium", "low", "base"]


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
    """

    max_tokens: int


class TLM:
    """TLM interface class."""

    def __init__(self, api_key: str, quality_preset: QualityPreset) -> None:
        """Initializes TLM interface."""
        self._api_key = api_key

        if quality_preset not in valid_quality_presets:
            raise ValueError(
                f"Invalid quality preset {quality_preset} -- must be one of {valid_quality_presets}"
            )

        self._quality_preset = quality_preset

    # TODO!
    # add batch_prompt, batch_prompt_async, batch_get_confidence_score, batch_get_confidence_score_async methods
    # share batch code between the _async methods (same rate limit, error handling)

    def prompt(self, prompt: str, options: Optional[TLMOptions] = None) -> TLMResponse:
        """
        Get response and confidence from TLM.

        Args:
            prompt (str): prompt for the TLM
        Returns:
            TLMResponse: [TLMResponse](#class-tlmresponse) object containing the response and confidence score
        """
        return asyncio.run(
            self.prompt_async(
                prompt,
                options,
            )
        )

    async def prompt_async(self, prompt: str, options: Optional[TLMOptions] = None) -> TLMResponse:
        """
        (Asynchronously) Get response and confidence from TLM.

        Args:
            prompt (str): prompt for the TLM
        Returns:
            TLMResponse: [TLMResponse](#class-tlmresponse) object containing the response and confidence score
        """
        tlm_response = await api.tlm_prompt(
            self._api_key,
            prompt,
            self._quality_preset,
            cast(JSONDict, options),
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
            response: response for the TLM  to evaluate
        Returns:
            float corresponding to the TLM's confidence score
        """
        return asyncio.run(
            self.get_confidence_score_async(
                prompt,
                response,
                options,
            )
        )

    async def get_confidence_score_async(
        self, prompt: str, response: str, options: Optional[TLMOptions] = None
    ) -> float:
        """(Asynchronously) gets confidence score for prompt-response pair.

        Args:
            prompt: prompt for the TLM
            response: response for the TLM  to evaluate
        Returns:
            float corresponding to the TLM's confidence score
        """
        if self._quality_preset == "base":
            raise ValueError(
                "Cannot get confidence score with `base` quality_preset -- choose a higher preset."
            )

        return cast(
            float,
            (
                await api.tlm_get_confidence_score(
                    self._api_key,
                    prompt,
                    response,
                    self._quality_preset,
                    cast(JSONDict, options),
                )
            )["confidence_score"],
        )
