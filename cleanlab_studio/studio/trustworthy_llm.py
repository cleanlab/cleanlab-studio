from typing import Tuple
from cleanlab_studio.internal.api import api


class TLM:
    """Handler for interactions with Trustworthy LLMs."""

    def __init__(self, api_key: str) -> None:
        """Initializes Trustworthy LLM hanlder w/ API key."""
        self._api_key = api_key

    def prompt(self, input: str) -> Tuple[str, float]:
        """
        Get inference and confidence from TLM.
        Args:
            input: question for the LLM
        Returns:
            A tuple of  the LLM output and a confidence score from [0,1]
        """
        response = api.get_tlm_confidence(self._api_key, input)
        return (response["answer"], response["confidence_score"])
