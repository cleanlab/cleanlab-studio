import abc
import io
from typing import Any, Awaitable, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from cleanlab_studio.internal.api import api


Predictions = npt.NDArray[np.int_] | npt.NDArray[np.str_]
ClassProbablities = pd.DataFrame


class Model(abc.ABC):
    """Base class for deployed model inference."""

    def __init__(self, api_key: str, model_id: str):
        """Initializes model class w/ API key and model ID."""
        self._api_key = api_key
        self._model_id = model_id

    @abc.abstractmethod
    def predict(
        self, batch: Any, return_pred_proba: bool = False
    ) -> Predictions | Tuple[Predictions, ClassProbablities]:
        """Gets predictions for batch of examples, optionally returning class probabilities.

        :param batch: batch of example to predict classes for
        :param return_pred_proba: if should return class probabilities, defaults to False
        :return: predictions + class probabilities, if requested
        """
        raise NotImplementedError

    def _predict(
        self, batch: io.StringIO, return_pred_proba: bool
    ) -> Predictions | Tuple[Predictions, ClassProbablities]:
        """Gets predictions for batch of examples, optionally returning class probabilities.

        :param batch: batch of example to predict classes for, as in-memory CSV file
        :param return_pred_proba: if should return class probabilities
        :return: predictions + class probabilities, if requested
        """
        return self._predict_async(batch, return_pred_proba)

    @abc.abstractmethod
    def predict_async(
        self, batch: Any, return_pred_proba: bool = False
    ) -> Awaitable[Predictions] | Awaitable[Tuple[Predictions, ClassProbablities]]:
        """Asynchronously gets predictions for batch of examples, optionally returning class probabilities.

        :param batch: batch of example to predict classes for
        :param return_pred_proba: if should return class probabilities, defaults to False
        :return: predictions + class probabilities, if requested
        """
        raise NotImplementedError

    def _predict_async(
        self, batch: io.StringIO, return_pred_proba: bool
    ) -> Predictions | Tuple[Predictions, ClassProbablities]:
        """Asynchronously gets predictions for batch of examples, optionally returning class probabilities.

        :param batch: batch of example to predict classes for, as in-memory CSV file
        :param return_pred_proba: if should return class probabilities, defaults to False
        :return: predictions + class probabilities, if requested
        """
        query_id: str = api.upload_predict_batch(self._api_key, self._model_id, batch)
        api.start_prediction(self._api_key, self._model_id, query_id)

        status: str | None = None
        result_url: str = ""
        while status != "done":
            status, result_url = api.get_prediction_status(
                self._api_key, query_id
            )

        # TODO handle get pred proba case
        return pd.read_csv(
            api.download_prediction_results(result_url),
        ).values
