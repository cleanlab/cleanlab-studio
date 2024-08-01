"""
Methods for interfacing with deployed ML models (to produce predictions).

**This module is not meant to be imported and used directly.** Instead, use [`Studio.get_model()`](../studio/#method-get_model) to instantiate a [Model](#class-model) object.

The [Model Deployment tutorial](/tutorials/inference_api/) explains the end-to-end workflow for using Cleanlab Studio's model deployment functionality.
"""

import abc
import csv
import io
import time
from typing import List, Optional, Tuple, Union, Any
from typing_extensions import TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

from cleanlab_studio.internal.api import api
from cleanlab_studio.errors import APITimeoutError


TextBatch = Union[List[str], npt.NDArray[np.str_], pd.Series]
TabularBatch: TypeAlias = pd.DataFrame
Batch = Union[TextBatch, TabularBatch]

Predictions = Union[npt.NDArray[np.int_], npt.NDArray[np.str_], npt.NDArray[Any]]
ClassProbablities: TypeAlias = pd.DataFrame


class Model(abc.ABC):
    """Represents a machine learning model instance in a Cleanlab Studio account.

    Models should be instantiated using the [`Studio.get_model()`](../studio/#method-get_model) method. Then, using a Model object, you can [`predict()`](#method-predict) labels for new data.
    """

    def __init__(self, api_key: str, model_id: str):
        """Initializes a model.

        **Objects of this class are not meant to be constructed directly.** Instead, use [`Studio.get_model()`](../studio/#method-get_model)."""
        self._api_key = api_key
        self._model_id = model_id

        model_info = api.get_deployed_model_info(self._api_key, self._model_id)
        self._tasktype = model_info["tasktype"]

    def predict(
        self,
        batch: Batch,
        return_pred_proba: bool = False,
        timeout: int = 930,  # default is 15 mins (lambdas time limit) + 30s
    ) -> Union[Predictions, Tuple[Predictions, ClassProbablities]]:
        """
        Gets predictions (and optionally, predicted probabilities) for batch of examples using deployed model.
                                Currently only supports tabular and text datasets.

        Args:
            batch: batch of examples to predict classes for
            return_pred_proba: whether to return predicted class probabilities for each example
            timeout: optional parameter to set timeout for predictions in seconds (defaults to 930s)

        Returns:
            Predictions: the predicted labels for the batch as a numpy array. For a multi-class model,
                                                returns a numpy array of the labels with types matching types of given labels in the
                                                original training set (string, integer, or boolean). For a multi-label model, returns a numpy array of lists of strings,
                                                where each list includes the labels that are present for the corresponding row.

            ClassProbabilities: optionally returns pandas DataFrame of the class probabilities where column names correspond to the labels.


        Example outputs:
            Multi-class project:
            Say we have a dataset with 3 classes, "bear", "cat" and "dog". For two example rows,
            the deployed model predicts "cat" and "dog", each with probability 1.
            The outputs will be:
                    Predictions: `array(['cat', 'dog'])`
                    ClassProbabilities:
                         bear  cat  dog
                    0    0.0   1.0  0.0
                    1    0.0   0.0  1.0
            Note that for multi-class predictions, ClassProbabilities rows will sum to 1,
            and the prediction for each row corresponds to the column with the highest probability.

            Multi-label project:
            Say we have some text dataset that we want sentiment predictions for, and the model predicts
            probabilities `[0.6, 0.9, 0.1]` for the set of possible labels, "happy", "excited", "sad"; for
            a second example, the predicted probabilities are `[0.1, 0.3, 0.8]`.
            The outputs will be:
                    Predictions: `array([["happy", "excited"], ["sad"]])`
                    ClassProbabilities:
                         happy  excited  sad
                    0    0.6    0.9      0.1
                    1    0.1    0.3      0.8
            Note that for multi-label predictions, each entry in a row of the ClassProbabilities should be interpreted
            as the probability that label is present for the example.
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
                if self._tasktype == "multi-label":
                    # convert multi-label suggested labels to list of list of strings
                    suggested_labels = (
                        results.pop("Suggested Label").apply(_csv_string_to_list).to_numpy()
                    )
                else:
                    suggested_labels = results.pop("Suggested Label").to_numpy()

                return suggested_labels, results

            time.sleep(1)

        else:
            raise APITimeoutError(f"Timeout of {timeout}s expired while waiting for prediction")

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


def _csv_string_to_list(csv_string: Optional[str]) -> List[str]:
    """Convert a csv string with one row that represents a list into a list

    Return empty list if string is empty
    """
    if pd.isna(csv_string):
        return []
    input_stream = io.StringIO(csv_string)
    reader = csv.reader(input_stream)
    for row in reader:
        return row
    return []
