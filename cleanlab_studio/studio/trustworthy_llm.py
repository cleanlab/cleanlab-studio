from typing import Literal, TypedDict
from cleanlab_studio.internal.api import api


QualityPreset = Literal["best", "high", "medium", "fast", "base"]


class TlmResponse(TypedDict):
    answer: str
    confidence_score: float


class TLM:
    """Handler for interactions with Trustworthy LLMs."""

    def __init__(self, api_key: str, quality_preset: QualityPreset) -> None:
        """Initializes Trustworthy LLM hanlder w/ API key."""
        self._api_key = api_key
        self._quality_preset = quality_preset

    def prompt(self, input: str) -> TlmResponse:
        """
        Get inference and confidence from TLM.

        Args:
            input: question for the LLM
        Returns:
            A dict containing the TLM response
        """
        response = api.tlm_prompt(self._api_key, input, self._quality_preset)
        return {
            "answer": response["answer"],
            "confidence_score": response["confidence_score"],
        }
