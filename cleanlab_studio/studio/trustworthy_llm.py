from typing import cast, Literal, TypedDict
from cleanlab_studio.internal.api import api


valid_quality_presets = ["best", "high", "medium", "low", "base"]
QualityPreset = Literal["best", "high", "medium", "low", "base"]


class TLMResponse(TypedDict):
    response: str
    confidence_score: float


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

    def prompt(self, prompt: str) -> TLMResponse:
        """
        Get inference and confidence from TLM.

        Args:
            prompt: prompt for the TLM
        Returns:
            A dict containing the TLM response
        """
        tlm_response = api.tlm_prompt(self._api_key, prompt, self._quality_preset)
        return {
            "response": tlm_response["response"],
            "confidence_score": tlm_response["confidence_score"],
        }

    def get_confidence_score(self, prompt: str, response: str) -> float:
        """Gets confidence score for prompt-response pair.

        Args:
            prompt: prompt for the TLM
            response: response for the TLM  to evaluate
        Returns:
            float corresponding to the TLM's confidence score
        """
        assert (
            self._quality_preset != "base"
        ), "Cannot get confidence score with `base` quality_preset -- choose a higher preset."
        return cast(
            float,
            api.tlm_get_confidence_score(self._api_key, prompt, response, self._quality_preset)[
                "confidence_score"
            ],
        )
