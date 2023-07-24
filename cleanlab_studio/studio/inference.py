import abc
import csv
import functools
import io
from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd

from cleanlab_studio.internal.api import api


TextBatch = List[str] | npt.NDArray[np.str_] | pd.Series
TabularBatch = pd.DataFrame
Batch = TextBatch | TabularBatch

Predictions = npt.NDArray[np.int_] | npt.NDArray[np.str_]
ClassProbablities = pd.DataFrame


class Model(abc.ABC):
    """Base class for deployed model inference."""

    def __init__(self, api_key: str, model_id: str):
        """Initializes model class w/ API key and model ID."""
        self._api_key = api_key
        self._model_id = model_id

    def predict(
        self,
        batch: Batch,
    ) -> Predictions:
        """Gets predictions for batch of examples.

        :param batch: batch of example to predict classes for
        :return: predictions from batch
        """
        csv_batch = self._convert_batch_to_csv(batch)
        return self._predict(csv_batch)

    def _predict(self, batch: io.StringIO) -> Predictions:
        """Gets predictions for batch of examples.

        :param batch: batch of example to predict classes for, as in-memory CSV file
        :return: predictions from batch
        """
        query_id: str = api.upload_predict_batch(self._api_key, self._model_id, batch)
        api.start_prediction(self._api_key, self._model_id, query_id)

        resp = api.get_prediction_status(self._api_key, query_id)
        status: str | None = resp["status"]
        while status != "done":
            resp = api.get_prediction_status(self._api_key, query_id)
            status = resp["status"]

        result_url = resp["result_url"]

        return pd.read_csv(
            api.download_prediction_results(result_url),
        ).values

    @functools.singledispatchmethod
    def _convert_batch_to_csv(self, batch: Batch) -> io.StringIO:
        """Converts batch object to CSV string IO."""
        sio = io.StringIO()

        # handle text batches
        if isinstance(batch, (list, np.ndarray, pd.Series)):
            writer = csv.writer(sio)

            # write header
            writer.writerow(["label"])

            # write labels to CSV
            for label in batch:
                writer.writerow([label])

        # handle tabular batches
        elif isinstance(batch, pd.DataFrame):
            batch.to_csv(sio)

        else:
            raise TypeError(f"Invalid type of batch: {type(batch)}")

        sio.seek(0)
        return sio
