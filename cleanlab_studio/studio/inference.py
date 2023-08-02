import abc
import csv
import io
import time
from typing import List, Tuple, Union
from typing_extensions import TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

from cleanlab_studio.internal.api import api


TextBatch = Union[List[str], npt.NDArray[np.str_], pd.Series]
TabularBatch: TypeAlias = pd.DataFrame
Batch = Union[TextBatch, TabularBatch]

Predictions = Union[npt.NDArray[np.int_], npt.NDArray[np.str_]]
ClassProbablities: TypeAlias = pd.DataFrame


class Model(abc.ABC):
    """Base class for deployed model inference."""

    def __init__(self, api_key: str, model_id: str):
        """Initializes model class w/ API key and model ID."""
        self._api_key = api_key
        self._model_id = model_id

    def predict(
        self,
        batch: Batch,
        return_pred_proba: bool = False,
        timeout: int = 600,
    ) -> Union[Predictions, Tuple[Predictions, ClassProbablities]]:
        """
        Gets predictions for batch of examples.

        Args:
            batch: batch of examples to predict classes for
            return_pred_proba: if should return class probabilities for each example
            timeout: optional parameter to set timeout for predictions in seconds

        Returns:
            predictions from batch as a numpy array, optionally also pandas dataframe of class probabilties
        """
        csv_batch = self._convert_batch_to_csv(batch)
        predictions, class_probabilities = self._predict_from_csv(csv_batch, timeout)

        if return_pred_proba:
            return predictions, class_probabilities

        return predictions

    def _predict_from_csv(
        self, batch: io.StringIO, timeout: int
    ) -> Tuple[Predictions, ClassProbablities]:
        """Gets predictions for batch of examples.

        :param batch: batch of example to predict classes for, as in-memory CSV file
        :return: predictions from batch, class probabilities
        """
        query_id: str = api.upload_predict_batch(self._api_key, self._model_id, batch)
        api.start_prediction(self._api_key, self._model_id, query_id)

        # Set timeout to prevent users from getting stuck indefinitely when there is a failure
        timeout_limit = time.time() + timeout
        while time.time() < timeout_limit:
            resp = api.get_prediction_status(self._api_key, query_id)

            if result_url := resp.get("results"):
                results: pd.DataFrame = pd.read_csv(result_url)
                return results.pop("Suggested Label").to_numpy(), results

            time.sleep(1)

        else:
            raise TimeoutError(f"Timeout of {timeout}s expired while waiting for prediction")

    @staticmethod
    def _convert_batch_to_csv(batch: Batch) -> io.StringIO:
        """Converts batch object to CSV string IO."""
        sio = io.StringIO()

        # handle text batches
        if isinstance(batch, (list, np.ndarray, pd.Series)):
            writer = csv.writer(sio)

            # write header
            writer.writerow(["text"])

            # write labels to CSV
            for input_data in batch:
                writer.writerow([input_data])

        # handle tabular batches
        elif isinstance(batch, pd.DataFrame):
            batch.to_csv(sio)

        else:
            raise TypeError(f"Invalid type of batch: {type(batch)}")

        sio.seek(0)
        return sio
