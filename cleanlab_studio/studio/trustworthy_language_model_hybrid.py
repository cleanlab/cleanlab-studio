from typing import List, Optional, Union, cast, Sequence, Any, Dict
import numpy as np

from .trustworthy_language_model import TLM, TLMOptions, TLMResponse
from cleanlab_studio.internal.tlm.validation import (
    validate_tlm_hybrid_score_options,
    get_tlm_hybrid_prompt_options,
)
from cleanlab_studio.internal.types import TLMQualityPreset
from cleanlab_studio.errors import ValidationError


class TLMHybrid:
    """
    A hybrid version of the Trustworthy Language Model (TLM) that enables

    Most of the input arguments for this class is similar to those for TLM

    Args:
        api_key: You can find your API key on your [account page](https://app.cleanlab.ai/account) in Cleanlab Studio. Instead of specifying the API key here, you can also log in with `cleanlab login` on the command-line.
    """

    def __init__(
        self,
        api_key: str,
        response_model: str,
        quality_preset: TLMQualityPreset,
        *,
        options: Optional[TLMOptions] = None,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = None,
    ) -> None:
        self._api_key = api_key
        self._response_model = response_model

        if quality_preset in {"high", "best"}:
            raise ValidationError(
                f"Invalid quality preset: {quality_preset}. TLMHybrid only supports 'low' and 'medium' presets."
            )
        self._score_quality_preset = quality_preset

        if options is not None:
            validate_tlm_hybrid_score_options(options)
        self._score_options = options
        self._prompt_options = cast(
            TLMOptions, get_tlm_hybrid_prompt_options(self._score_options, self._response_model)
        )

        self._timeout = timeout  # how to handle, half?
        self._verbose = verbose

        self.tlm_response = TLM(
            self._api_key,
            quality_preset="base",
            options=self._prompt_options,
            timeout=self._timeout,
            verbose=self._verbose,
        )

        self.tlm_score = TLM(
            self._api_key,
            quality_preset=self._score_quality_preset,
            options=self._score_options,
            timeout=self._timeout,
            verbose=self._verbose,
        )

    def prompt(
        self,
        prompt: Union[str, Sequence[str]],
    ) -> Union[TLMResponse, List[TLMResponse]]:
        prompt_response = self.tlm_response.prompt(prompt)

        if isinstance(prompt_response, list):
            assert isinstance(prompt, Sequence)
            response = [r["response"] for r in prompt_response]
            perplexity = [r["log"]["perplexity"] for r in prompt_response]
            return self._batch_score(prompt, response, perplexity)

        assert isinstance(prompt, str)
        return self._score(
            prompt,
            prompt_response["response"],
            perplexity=prompt_response["log"]["perplexity"],
        )

    def try_prompt(
        self,
        prompt: Sequence[str],
    ) -> List[Optional[TLMResponse]]:
        prompt_response = self.tlm_response.try_prompt(prompt)
        prompt_succeeded_mask = [res is not None for res in prompt_response]

        if not any(prompt_succeeded_mask):  # all prompts failed
            return [None] * len(prompt)

        # handle masking with numpy for easier indexing
        prompt_succeeded = np.array(prompt)[prompt_succeeded_mask].tolist()
        prompt_response_succeeded = np.array(prompt_response)[prompt_succeeded_mask]

        response_succeeded = [r["response"] for r in prompt_response_succeeded]
        perplexity_succeeded = [r["log"]["perplexity"] for r in prompt_response_succeeded]
        score_response_succeeded = self._try_batch_score(
            prompt_succeeded, response_succeeded, perplexity_succeeded
        )

        tlm_response = np.full(len(prompt), None)
        tlm_response[prompt_succeeded_mask] = score_response_succeeded

        return tlm_response.tolist()

    def _score(
        self,
        prompt: str,
        response: str,
        perplexity: Optional[float],
    ) -> TLMResponse:
        score_response = self.tlm_score.get_trustworthiness_score(
            prompt, response, perplexity=perplexity
        )

        if isinstance(score_response, dict):
            return {"response": response, **score_response}
        elif isinstance(score_response, (float, int)):
            return {"response": response, "trustworthiness_score": score_response}
        else:
            raise ValueError(f"score_response has invalid type {type(score_response)}")

    def _batch_score(
        self,
        prompt: Sequence[str],
        response: Sequence[str],
        perplexity: Sequence[Optional[float]],
    ) -> List[TLMResponse]:
        score_response = self.tlm_score.get_trustworthiness_score(
            prompt, response, perplexity=perplexity
        )

        assert isinstance(score_response, list)
        assert len(prompt) == len(score_response)

        if all(isinstance(score, dict) for score in score_response):
            return [{"response": res, **score} for res, score in zip(response, score_response)]

        return [
            {"response": res, "trustworthiness_score": score}
            for res, score in zip(response, score_response)
        ]

    def _try_batch_score(
        self,
        prompt: Sequence[str],
        response: Sequence[str],
        perplexity: Sequence[Optional[float]],
    ) -> List[TLMResponse]:
        score_response = self.tlm_score.try_get_trustworthiness_score(
            prompt, response, perplexity=perplexity
        )

        assert len(prompt) == len(score_response)

        if all(isinstance(score, dict) for score in score_response):
            return [
                {"response": res, **score} if score else None
                for res, score in zip(response, score_response)
            ]

        return [
            {"response": res, "trustworthiness_score": score} if score else None
            for res, score in zip(response, score_response)
        ]
