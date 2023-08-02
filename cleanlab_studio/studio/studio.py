"""
Python API for Cleanlab Studio.
"""
from typing import Any, List, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import clean, upload, inference
from cleanlab_studio.internal.api import api
from cleanlab_studio.internal.util import (
    init_dataset_source,
    check_none,
    check_not_none,
)
from cleanlab_studio.internal.settings import CleanlabSettings
from cleanlab_studio.internal.types import FieldSchemaDict

_pyspark_exists = api.pyspark_exists
if _pyspark_exists:
    import pyspark.sql


class Studio:
    """Used to interact with Cleanlab Studio."""

    _api_key: str

    def __init__(self, api_key: Optional[str]):
        if not api.is_valid_client_version():
            raise ValueError(
                "CLI is out of date and must be updated. Run 'pip install --upgrade cleanlab-studio'."
            )
        if api_key is None:
            try:
                api_key = CleanlabSettings.load().api_key
                if api_key is None:
                    raise ValueError
            except (FileNotFoundError, KeyError, ValueError):
                raise ValueError(
                    "No API key found; either specify API key or log in with 'cleanlab login' first"
                )
        api.validate_api_key(api_key)
        self._api_key = api_key
        self.experimental = self.Experimental(self)  # type: ignore

    def upload_dataset(
        self,
        dataset: Any,
        dataset_name: Optional[str] = None,
        *,
        schema_overrides: Optional[FieldSchemaDict] = None,
        modality: Optional[str] = None,
        id_column: Optional[str] = None,
    ) -> str:
        """
        Uploads a dataset to Cleanlab Studio.

        Args:
            dataset: Object representing the dataset to upload. Currently supported formats include a `str` path to your dataset, a pandas DataFrame, a pyspark DataFrame.
            dataset_name: Name for your dataset in Cleanlab Studio (optional if uploading from filepath).
            schema_overrides: Optional dictionary of overrides you would like to make to the schema of your dataset. If not provided, schema will be inferred.
            modality: Optional parameter to override the modality of your dataset. If not provided, modality will be inferred.
            id_column: Optional parameter to override the ID column of your dataset. If not provided, a monotonically increasing ID column will be generated.

        Returns:
            ID of uploaded dataset.
        """
        ds = init_dataset_source(dataset, dataset_name)
        return upload.upload_dataset(
            self._api_key,
            ds,
            schema_overrides=schema_overrides,
            modality=modality,
            id_column=id_column,
        )

    def download_cleanlab_columns(
        self,
        cleanset_id: str,
        include_action: bool = False,
        to_spark: bool = False,
    ) -> Any:
        """
        Downloads Cleanlab columns for a cleanset.

        Args:
            cleanset_id: ID of cleanset to download columns from. To obtain cleanset ID from project ID use, [get_latest_cleanset_id](#method-get_latest_cleanset_id).
            include_action: Whether to include a column with any actions taken on the cleanset in the downloaded columns.

        Returns:
            A pandas or pyspark DataFrame. Type is `Any` to avoid requiring pyspark installation.
        """
        rows_df = api.download_cleanlab_columns(
            self._api_key, cleanset_id, all=True, to_spark=to_spark
        )
        if not include_action:
            if to_spark:
                rows_df = rows_df.drop("action")
            else:
                rows_df.drop("action", inplace=True, axis=1)
        return rows_df

    def apply_corrections(self, cleanset_id: str, dataset: Any, keep_excluded: bool = False) -> Any:
        """
        Applies corrections from a Cleanlab Studio cleanset to your dataset. Corrections can be made by viewing your project in the Cleanlab Studio webapp (see [Cleanlab Studio web quickstart](/guide/quickstart/web#review-the-errors)).

        Args:
            cleanset_id: ID of cleanset to apply corrections from.
            dataset: Dataset to apply corrections to. Supported formats include pandas DataFrame and pyspark DataFrame. Dataset should have the same number of rows as the dataset used to create the project. It should also contain a label column with the same name as the label column for the project.
            keep_excluded: Whether to retain rows with an "exclude" action. By default these rows will be removed from the dataset.

        Returns:
            A copy of the dataset with corrections applied.
        """
        project_id = api.get_project_of_cleanset(self._api_key, cleanset_id)
        label_column = api.get_label_column_of_project(self._api_key, project_id)
        id_col = api.get_id_column(self._api_key, cleanset_id)
        if _pyspark_exists and isinstance(dataset, pyspark.sql.DataFrame):
            from pyspark.sql.functions import udf

            cl_cols = self.download_cleanlab_columns(
                cleanset_id, include_action=True, to_spark=True
            )
            corrected_ds_spark = dataset.alias("corrected_ds")
            if id_col not in corrected_ds_spark.columns:
                from pyspark.sql.functions import (
                    row_number,
                    monotonically_increasing_id,
                )
                from pyspark.sql.window import Window

                corrected_ds_spark = corrected_ds_spark.withColumn(
                    id_col,
                    row_number().over(Window.orderBy(monotonically_increasing_id())) - 1,
                )
            both = cl_cols.select([id_col, "action", "clean_label"]).join(
                corrected_ds_spark.select([id_col, label_column]),
                on=id_col,
                how="left",
            )
            final = both.withColumn(
                "__cleanlab_final_label",
                # XXX hacky, checks if label is none by hand
                # instead, use original JSON, which uses null values where it's not specified
                udf(lambda original, clean: original if check_none(clean) else clean)(
                    both[label_column],
                    "clean_label",
                ),
            )
            new_labels = final.select(
                [id_col, "action", "__cleanlab_final_label"]
            ).withColumnRenamed("__cleanlab_final_label", label_column)
            return (
                corrected_ds_spark.drop(label_column)
                .join(new_labels, on=id_col, how="right")
                .where(new_labels["action"] != "exclude")
                .drop("action")
            )
        elif isinstance(dataset, pd.DataFrame):
            cl_cols = self.download_cleanlab_columns(cleanset_id, include_action=True)
            joined_ds: pd.DataFrame
            if id_col in dataset.columns:
                joined_ds = dataset.join(cl_cols.set_index(id_col), on=id_col)
            else:
                joined_ds = dataset.join(cl_cols.set_index(id_col).sort_values(by=id_col))
            joined_ds["__cleanlab_final_label"] = joined_ds["clean_label"].where(
                np.asarray(list(map(check_not_none, joined_ds["clean_label"].to_numpy()))),
                dataset[label_column].to_numpy(),
            )

            corrected_ds: pd.DataFrame = dataset.copy()
            corrected_ds[label_column] = joined_ds["__cleanlab_final_label"]
            if not keep_excluded:
                corrected_ds = corrected_ds.loc[(joined_ds["action"] != "exclude").fillna(True)]
            else:
                corrected_ds["action"] = joined_ds["action"]
            return corrected_ds

        else:
            raise ValueError(
                f"Provided unsupported dataset of type: {type(dataset)}. We currently support applying corrections to pandas or pyspark dataframes"
            )

    def create_project(
        self,
        dataset_id: str,
        project_name: str,
        modality: Literal["text", "tabular", "image"],
        *,
        task_type: Literal["multi-class", "multi-label"] = "multi-class",
        model_type: Literal["fast", "regular"] = "regular",
        label_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        text_column: Optional[str] = None,
    ) -> str:
        """
        Creates a Cleanlab Studio project.

        Args:
            dataset_id: ID of dataset to create project for.
            project_name: Name for resulting project.
            modality: Modality of project (i.e. text, tabular, image).
            task_type: Type of classification to perform (i.e. multi-class, multi-label).
            model_type: Type of model to train (i.e. fast, regular).
            label_column: Name of column in dataset containing labels (if not supplied, we'll make our best guess).
            feature_columns: List of columns to use as features when training tabular modality project (if not supplied and modality is "tabular" we'll use all valid feature columns).
            text_column: Name of column containing the text to train text modality project on (if not supplied and modality is "text" we'll make our best guess).

        Returns:
            ID of created project.
        """
        dataset_details = api.get_dataset_details(self._api_key, dataset_id)

        if label_column is not None:
            if label_column not in dataset_details["label_columns"]:
                raise ValueError(
                    f"Invalid label column: {label_column}. Label column must have categorical feature type"
                )
        else:
            label_column = str(dataset_details["label_column_guess"])
            print(f"Label column not supplied. Using best guess {label_column}")

        if feature_columns is not None and modality != "tabular":
            if label_column in feature_columns:
                raise ValueError("Label column cannot be included in feature columns")
            raise ValueError("Feature columns supplied, but project modality is not tabular")
        if feature_columns is None:
            if modality == "tabular":
                feature_columns = dataset_details["distinct_columns"]
                feature_columns.remove(label_column)
                print(f"Feature columns not supplied. Using all valid feature columns")

        if text_column is not None:
            if modality != "text":
                raise ValueError("Text column supplied, but project modality is not text")
            elif text_column not in dataset_details["text_columns"]:
                raise ValueError(
                    f"Invalid text column: {text_column}. Column must have text feature type"
                )
        if text_column is None and modality == "text":
            text_column = dataset_details["text_column_guess"]
            print(f"Text column not supplied. Using best guess {text_column}")

        return api.clean_dataset(
            api_key=self._api_key,
            dataset_id=dataset_id,
            project_name=project_name,
            task_type=task_type,
            modality=modality,
            model_type=model_type,
            label_column=label_column,
            feature_columns=feature_columns if feature_columns is not None else [],
            text_column=text_column,
        )

    def poll_cleanset_status(self, cleanset_id: str, timeout: Optional[int] = None) -> bool:
        """
        Repeatedly polls for cleanset status while the cleanset is being generated. Blocks until cleanset is ready, there is a cleanset error, or `timeout` is exceeded.

        Args:
            cleanset_id: ID of cleanset to check status of.
            timeout: Optional timeout after which to stop polling for progress. If not provided, will block until cleanset is ready.

        Returns:
            After cleanset is done being generated, returns `True` if cleanset is ready to use, `False` otherwise.
        """
        return clean.poll_cleanset_status(self._api_key, cleanset_id, timeout)

    def get_latest_cleanset_id(self, project_id: str) -> str:
        """
        Gets latest cleanset ID for a project.

        Args:
            project_id: ID of project.

        Returns:
            ID of latest associated cleanset.
        """
        return api.get_latest_cleanset_id(self._api_key, project_id)

    def delete_project(self, project_id: str) -> None:
        """
        Deletes a project from Cleanlab Studio.

        Args:
            project_id: ID of project to delete.
        """
        api.delete_project(self._api_key, project_id)
        print(f"Successfully deleted project: {project_id}")

    def get_model(self, model_id: str) -> inference.Model:
        """
        Gets a model deployed by Cleanlab Studio.

        Args:
            model_id: ID of model to get. This ID should be fetched in the deployments page of the app UI.

        Returns:
            Model object with methods to run predictions on new input data
        """
        return inference.Model(self._api_key, model_id)

    class Experimental:
        def __init__(self, outer):  # type: ignore
            self._outer = outer

        def download_pred_probs(
            self,
            cleanset_id: str,
        ) -> Union[npt.NDArray[np.float_], pd.DataFrame]:
            """
            Downloads predicted probabilities for a cleanset
            Old pred_probs were saved as numpy arrays, which is still compatible
            Newer pred_probs are saved as pd.DataFrames
            """
            return api.download_array(self._outer._api_key, cleanset_id, "pred_probs")

        def download_embeddings(
            self,
            cleanset_id: str,
        ) -> Union[npt.NDArray[np.float_], pd.DataFrame]:
            """
            Downloads embeddings for a cleanset
            The downloaded array will always be a numpy array, the above is just for typing purposes
            """
            return api.download_array(self._outer._api_key, cleanset_id, "embeddings")
