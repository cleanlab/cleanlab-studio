"""
Cleanlab TLM is a Large Language Model that gives more reliable answers and quantifies its uncertainty in these answers
"""
from typing import cast, Literal, Optional, TypedDict
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

    max_tokens: int
    max_timeout: int
    num_candidate_responses: int
    num_consistency_samples: int
    use_self_reflection: bool


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

    def prompt(self, prompt: str, options: Optional[TLMOptions] = None) -> TLMResponse:
        """
        Get response and confidence from TLM.

        Args:
            prompt (str): prompt for the TLM
        Returns:
            TLMResponse: [TLMResponse](#class-tlmresponse) object containing the response and confidence score
        """
        tlm_response = api.tlm_prompt(
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
        if self._quality_preset == "base":
            raise ValueError(
                "Cannot get confidence score with `base` quality_preset -- choose a higher preset."
            )

        return cast(
            float,
            api.tlm_get_confidence_score(
                self._api_key,
                prompt,
                response,
                self._quality_preset,
                cast(JSONDict, options),
            )["confidence_score"],
        )
