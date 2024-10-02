"""
TLM Calibrated is a variant of the Trustworthy Language Model (TLM) that facilitates the calibration of trustworthiness scores
using existing ratings for prompt-response pairs, which allows for better alignment of the TLM scores in specialized-use cases.

**This module is not meant to be imported and used directly.**
Instead, use [`Studio.TLMCalibrated()`](/reference/python/studio/#method-tlmcalibrated) to instantiate a [TLMCalibrated](#class-tlmcalibrated) object,
and then you can use the methods like [`get_trustworthiness_score()`](#method-get_trustworthiness_score) documented on this page.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from cleanlab_studio.errors import TlmNotCalibratedError, ValidationError
from cleanlab_studio.internal.types import TLMQualityPreset
from cleanlab_studio.studio.trustworthy_language_model import TLM, TLMOptions, TLMResponse, TLMScore


class TLMCalibrated:
    def __init__(
        self,
        api_key: str,
        quality_preset: TLMQualityPreset,
        *,
        options: Optional[TLMOptions] = None,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = None,
    ) -> None:
        """
        Use `Studio.TLMCalibrated()` instead of this method to initialize a TLMCalibrated object.
        lazydocs: ignore
        """
        try:
            from sklearn.ensemble import RandomForestRegressor  # type: ignore
        except ImportError:
            raise ImportError(
                "Cannot import scikit-learn which is required to use TLMCalibrated. "
                "Please install it using `pip install scikit-learn` and try again."
            )

        self._api_key = api_key

        if quality_preset not in {"base", "low", "medium"}:
            raise ValidationError(
                f"Invalid quality preset: {quality_preset}. TLMCalibrated only supports 'base', 'low' and 'medium' presets."
            )
        self._quality_preset = quality_preset

        self._options = options
        self._timeout = timeout if timeout is not None and timeout > 0 else None
        self._verbose = verbose

        custom_eval_criteria_list = (
            self._options.get("custom_eval_criteria", []) if self._options else []
        )

        # number of custom eval critera + 1 to account for the default TLM trustworthiness score
        self._num_features = len(custom_eval_criteria_list) + 1
        self._rf_model = RandomForestRegressor(monotonic_cst=[1] * self._num_features)

        self._tlm = TLM(
            self._api_key,
            quality_preset=self._quality_preset,
            options=self._options,
            timeout=self._timeout,
            verbose=self._verbose,
        )

    def fit(self, tlm_scores: List[TLMScore], ratings: Sequence[float]) -> None:
        """
        Callibrate the model using TLM scores obtained from a previous `TLM.get_trustworthiness_score()` call
        using the provided numeric ratings.

        Args:
            tlm_scores (List[TLMScore]): list of [TLMScore](../trustworthy_language_model/#class-tlmscore) object obtained
                from a previous `TLM.get_trustworthiness_score()` call
            ratings (Sequence[float]): sequence of numeric ratings corresponding to each prompt-response pair,
                the length of this sequence must match the length of the `tlm_scores`.
        """
        if len(tlm_scores) != len(ratings):
            raise ValidationError(
                "The list of ratings must be of the same length as the list of TLM scores."
            )

        tlm_scores_df = pd.DataFrame(tlm_scores)
        extracted_scores = self._extract_tlm_scores(tlm_scores_df)

        if extracted_scores.shape[1] != self._num_features:
            raise ValidationError(
                f"TLMCalibrated has {self._num_features - 1} custom evaluation criteria defined, "
                f"however the tlm_scores provided have {extracted_scores.shape[1] - 1} custom evaluation scores. "
                "Please make sure the number of custom evaluation criterias match."
            )

        # using pandas so that NaN values are handled correctly
        ratings_series = pd.Series(ratings)
        ratings_normalized = (ratings_series - ratings_series.min()) / (
            ratings_series.max() - ratings_series.min()
        )

        self._rf_model.fit(extracted_scores, ratings_normalized.values)

    def prompt(
        self, prompt: Union[str, Sequence[str]]
    ) -> Union[TLMResponseWithCalibration, List[TLMResponseWithCalibration]]:
        """
        Gets response and a calibrated trustworthiness score for the given prompts,
        make sure that the model has been calibrated by calling the `.fit()` method before using this method.

        Similar to [`TLM.prompt()`](../trustworthy_language_model/#method-prompt),
        view documentation there for expected input arguments and outputs.
        """
        try:
            from sklearn.exceptions import NotFittedError  # type: ignore
            from sklearn.utils.validation import check_is_fitted  # type: ignore
        except ImportError:
            raise ImportError(
                "Cannot import scikit-learn which is required to use TLMCalibrated. "
                "Please install it using `pip install scikit-learn` and try again."
            )

        try:
            check_is_fitted(self._rf_model)
        except NotFittedError:
            raise TlmNotCalibratedError(
                "TLMCalibrated has to be calibrated before prompting new data, use the .fit() method to calibrate the model."
            )

        tlm_response = self._tlm.prompt(prompt)

        is_single_query = isinstance(tlm_response, dict)
        if is_single_query:
            assert not isinstance(tlm_response, list)
            tlm_response = [tlm_response]
        tlm_response_df = pd.DataFrame(tlm_response)

        extracted_scores = self._extract_tlm_scores(tlm_response_df)

        tlm_response_df["calibrated_score"] = self._rf_model.predict(extracted_scores)

        if is_single_query:
            return cast(TLMResponseWithCalibration, tlm_response_df.to_dict(orient="records")[0])

        return cast(List[TLMResponseWithCalibration], tlm_response_df.to_dict(orient="records"))

    def get_trustworthiness_score(
        self, prompt: Union[str, Sequence[str]], response: Union[str, Sequence[str]]
    ) -> Union[TLMScoreWithCalibration, List[TLMScoreWithCalibration]]:
        """
        Computes the calibrated trustworthiness score for arbitrary given prompt-response pairs,
        make sure that the model has been calibrated by calling the `.fit()` method before using this method.

        Similar to [`TLM.get_trustworthiness_score()`](../trustworthy_language_model/#method-get_trustworthiness_score),
        view documentation there for expected input arguments and outputs.
        """
        try:
            from sklearn.exceptions import NotFittedError
            from sklearn.utils.validation import check_is_fitted
        except ImportError:
            raise ImportError(
                "Cannot import scikit-learn which is required to use TLMCalibrated. "
                "Please install it using `pip install scikit-learn` and try again."
            )

        try:
            check_is_fitted(self._rf_model)
        except NotFittedError:
            raise TlmNotCalibratedError(
                "TLMCalibrated has to be calibrated before scoring new data, use the .fit() method to calibrate the model."
            )

        tlm_scores = self._tlm.get_trustworthiness_score(prompt, response)

        is_single_query = isinstance(tlm_scores, dict)
        if is_single_query:
            assert not isinstance(tlm_scores, list)
            tlm_scores = [tlm_scores]
        tlm_scores_df = pd.DataFrame(tlm_scores)

        extracted_scores = self._extract_tlm_scores(tlm_scores_df)

        tlm_scores_df["calibrated_score"] = self._rf_model.predict(extracted_scores)

        if is_single_query:
            return cast(TLMScoreWithCalibration, tlm_scores_df.to_dict(orient="records")[0])

        return cast(List[TLMScoreWithCalibration], tlm_scores_df.to_dict(orient="records"))

    def _extract_tlm_scores(self, tlm_scores_df: pd.DataFrame) -> npt.NDArray[np.float64]:
        """
        Transform a DataFrame containing TLMScore objects into a 2D numpy array,
        where each column represents different scores including trustworthiness score and any custom evaluation criteria.

        Args:
            tlm_scores_df: DataFrame constructed using a list of TLMScore objects.

        Returns:
            np.ndarray: 2D numpy array where each column corresponds to different scores.
              The first column is the trustworthiness score, followed by any custom evaluation scores if present.
        """
        tlm_log = tlm_scores_df.get("log", None)

        # if custom_eval_criteria is present in the log, use it as features
        if tlm_log is not None and "custom_eval_criteria" in tlm_log.iloc[0]:
            custom_eval_scores = np.array(
                tlm_scores_df["log"]
                .apply(lambda x: [criteria["score"] for criteria in x["custom_eval_criteria"]])
                .tolist()
            )
            all_scores = np.hstack(
                [tlm_scores_df["trustworthiness_score"].values.reshape(-1, 1), custom_eval_scores]
            )
        # otherwise use the TLM trustworthiness score as the only feature
        else:
            all_scores = tlm_scores_df["trustworthiness_score"].values.reshape(-1, 1)

        return all_scores


class TLMResponseWithCalibration(TLMResponse):
    """
    A typed dict similar to [TLMResponse](../trustworthy_language_model/#class-tlmresponse) but containing an extra key `calibrated_score`.
    View [TLMResponse](../trustworthy_language_model/#class-tlmresponse) for the description of the other keys in this dict.

    Attributes:
        calibrated_score (float, optional): score between 0 and 1 that has been calibrated to the provided ratings.
        A higher score indicates a higher confidence that the response is correct/trustworthy.
    """

    calibrated_score: Optional[float]


class TLMScoreWithCalibration(TLMScore):
    """
    A typed dict similar to [TLMScore](../trustworthy_language_model/#class-tlmscore) but containing an extra key `calibrated_score`.
    View [TLMScore](../trustworthy_language_model/#class-tlmscore) for the description of the other keys in this dict.

    Attributes:
        calibrated_score (float, optional): score between 0 and 1 that has been calibrated to the provided ratings.
        A higher score indicates a higher confidence that the response is correct/trustworthy.
    """

    calibrated_score: Optional[float]
