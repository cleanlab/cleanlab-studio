"""
TLMHybrid is a hybrid version of the [Trustworthy Language Model (TLM)](../trustworthy_language_model) that enables the use of different LLMs for generating the response and for scoring its trustworthiness.

**This module is not meant to be imported and used directly.** Instead, use [`Studio.TLMHybrid()`](/reference/python/studio/#method-tlmhybrid) to instantiate a [TLMHybrid](#class-tlmhybrid) object, and then you can use the methods like [`prompt()`](#method-prompt) documented on this page.
"""

from typing import List, Optional, Union, cast, Sequence
import numpy as np

from cleanlab_studio.errors import ValidationError
from cleanlab_studio.internal.tlm.validation import (
    validate_tlm_hybrid_score_options,
    get_tlm_hybrid_response_options,
)
from cleanlab_studio.internal.types import TLMQualityPreset
from cleanlab_studio.studio.trustworthy_language_model import TLM, TLMOptions, TLMResponse, TLMScore


class TLMHybrid:
    """
    A hybrid version of the Trustworthy Language Model (TLM) that enables the use of different LLMs for generating the response and for scoring its trustworthiness.

    TLMHybrid should be used if you want to use a better model to generate responses but want to get cheaper and quicker trustworthiness score
    evaluations by using smaller models.

    ** The TLMHybrid object is not meant to be constructed directly.** Instead, use the [`Studio.TLMHybrid()`](../studio/#method-tlmhybrid)
    method to configure and instantiate a TLMHybrid object.
    After you've instantiated the TLMHybrid object using [`Studio.TLMHybrid()`](../studio/#method-tlmhybrid), you can use the instance methods documented on this page.
    Possible arguments for `Studio.TLMHybrid()` are documented below.

    Most of the input arguments for this class are similar to those for TLM, major differences will be described below.

    Args:
        response_model (str): LLM used to produce the response to the given prompt.
            Do not specify the model to use for scoring  trustworthiness here, instead specify that model in the `options` argument.
            The list of supported model strings can be found in the TLMOptions documentation, by default, the model is "gpt-4o".

        quality_preset (TLMQualityPreset, default = "medium"): preset configuration to control the quality of TLM trustworthiness scores vs. runtimes/costs.
            This preset only applies to the model computing the trustworthiness score.
            Supported options are only "medium" or "low", because TLMHybrid is not intended to improve response accuracy
            (use the regular [TLM](../trustworthy_language_model) for that).

        options (TLMOptions, optional): a typed dict of advanced configuration options.
            Most of these options only apply to the model scoring  trustworthiness, except for "max_tokens", which applies to the response model as well.
            Specify which model to use for scoring trustworthiness in these options.
            For more details about the options, see the documentation for [TLMOptions](../trustworthy_language_model/#class-tlmoptions).

        timeout (float, optional): timeout (in seconds) to apply to each TLM prompt.

        verbose (bool, optional): whether to print outputs during execution, i.e., whether to show a progress bar when TLM is prompted with batches of data.
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
        """
        Use `Studio.TLMHybrid()` instead of this method to initialize a TLMHybrid object.
        lazydocs: ignore
        """
        self._api_key = api_key
        self._response_model = response_model

        if quality_preset not in {"low", "medium"}:
            raise ValidationError(
                f"Invalid quality preset: {quality_preset}. TLMHybrid only supports 'low' and 'medium' presets."
            )
        self._score_quality_preset = quality_preset

        if options is not None:
            validate_tlm_hybrid_score_options(options)
        self._score_options = options
        self._response_options = cast(
            TLMOptions, get_tlm_hybrid_response_options(self._score_options, self._response_model)
        )

        self._timeout = (
            timeout if timeout is not None and timeout > 0 else None
        )  # TODO: better timeout handling
        self._verbose = verbose

        self._tlm_response = TLM(
            self._api_key,
            quality_preset="base",
            options=self._response_options,
            timeout=self._timeout,
            verbose=self._verbose,
        )

        self._tlm_score = TLM(
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
        """
        Similar to [`TLM.prompt()`](../trustworthy_language_model/#method-prompt), view documentation there for expected input arguments and outputs.
        """
        prompt_response = self._tlm_response.prompt(prompt)

        if isinstance(prompt, Sequence) and isinstance(prompt_response, list):
            response = [r["response"] for r in prompt_response]
            perplexity = [r["log"]["perplexity"] for r in prompt_response]
            return self._batch_score(prompt, response, perplexity)

        elif isinstance(prompt, str) and isinstance(prompt_response, dict):
            return self._score(
                prompt,
                prompt_response["response"],
                perplexity=prompt_response["log"]["perplexity"],
            )

        else:
            raise ValueError(f"prompt and prompt_response do not have matching types")

    def try_prompt(
        self,
        prompt: Sequence[str],
    ) -> List[Optional[TLMResponse]]:
        """
        Similar to [`TLM.try_prompt()`](../trustworthy_language_model/#method-try_prompt), view documentation there for expected input arguments and outputs.
        """
        prompt_response = self._tlm_response.try_prompt(prompt)
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

        return cast(List[Optional[TLMResponse]], tlm_response.tolist())

    def _score(
        self,
        prompt: str,
        response: str,
        perplexity: Optional[float],
    ) -> TLMResponse:
        """
        Private method to get trustworthiness score for a single example and process the outputs into a TLMResponse dictionary.

        Args:
            prompt: prompt for the TLM to evaluate
            response: response corresponding to the input prompt
            perplexity: perplexity of the response (given by LLM)
        Returns:
            TLMResponse: [TLMResponse](#class-tlmresponse) object containing the response and trustworthiness score
        """
        score_response = self._tlm_score.get_trustworthiness_score(
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
        """
        Private method to get trustworthiness score for a batch of examples and process the outputs into TLMResponse dictionaries.

        Args:
            prompt: list of prompts for the TLM to evaluate
            response: list of responses corresponding to the input prompt
            perplexity: list of perplexity scores of the response (given by LLM)
        Returns:
            List[TLMResponse]: list of [TLMResponse](#class-tlmresponse) object containing the response and trustworthiness score
        """
        score_response = self._tlm_score.get_trustworthiness_score(
            prompt, response, perplexity=perplexity
        )

        assert isinstance(score_response, list)
        assert len(prompt) == len(score_response)

        if all(isinstance(score, dict) for score in score_response):
            score_response = cast(List[TLMScore], score_response)
            return [{"response": res, **score} for res, score in zip(response, score_response)]
        elif all(isinstance(score, (float, int)) for score in score_response):
            score_response = cast(List[float], score_response)
            return [
                {"response": res, "trustworthiness_score": score}
                for res, score in zip(response, score_response)
            ]
        else:
            raise ValueError(f"score_response has invalid type")

    def _try_batch_score(
        self,
        prompt: Sequence[str],
        response: Sequence[str],
        perplexity: Sequence[Optional[float]],
    ) -> List[Optional[TLMResponse]]:
        """
        Private method to get trustworthiness score for a batch of examples and process the outputs into TLMResponse dictionaries,
        handling any failures (errors of timeouts) by returning None in place of the failures.

        Args:
            prompt: list of prompts for the TLM to evaluate
            response: list of responses corresponding to the input prompt
            perplexity: list of perplexity scores of the response (given by LLM)
        Returns:
            List[TLMResponse]: list of [TLMResponse](#class-tlmresponse) object containing the response and trustworthiness score.
                In case of any failures, the return list will contain None in place of the TLM response for that example.
        """
        score_response = self._tlm_score.try_get_trustworthiness_score(
            prompt, response, perplexity=perplexity
        )

        assert len(prompt) == len(score_response)

        if all(score is None or isinstance(score, dict) for score in score_response):
            score_response = cast(List[Optional[TLMScore]], score_response)
            return [
                {"response": res, **score} if score else None
                for res, score in zip(response, score_response)
            ]
        elif all(score is None or isinstance(score, (float, int)) for score in score_response):
            score_response = cast(List[Optional[float]], score_response)
            return [
                {"response": res, "trustworthiness_score": score} if score else None
                for res, score in zip(response, score_response)
            ]
        else:
            raise ValueError(f"score_response has invalid type")
