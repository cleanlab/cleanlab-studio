from typing import TypedDict
from cleanlab_studio.internal.api import api


class TlmResponse(TypedDict):
    answer: str
    confidence_score: float


class TLM:
    """Handler for interactions with Trustworthy LLMs."""

    def __init__(self, api_key: str) -> None:
        """Initializes Trustworthy LLM hanlder w/ API key."""
        self._api_key = api_key

    def prompt(self, input: str) -> TlmResponse:
        """
        Get inference and confidence from TLM.

        Args:
            input: question for the LLM
        Returns:
            A dict containing the TLM response
        """
        response = api.get_tlm_confidence(self._api_key, input)
        return {
            "answer": response["answer"],
            "confidence_score": response["confidence_score"],
        }
