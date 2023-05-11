from typing import Any, List, Literal, Optional

import numpy as np
import pandas as pd

try:
    import pyspark.sql

    pyspark_exists = True
except ImportError:
    pyspark_exists = False

from . import clean, upload
from cleanlab_studio.internal.api import api
from cleanlab_studio.internal.util import init_dataset_source, as_numpy_type
from cleanlab_studio.internal.settings import CleanlabSettings
from cleanlab_studio.internal.types import FieldSchemaDict


class Studio:
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

    def upload_dataset(
        self,
        dataset: Any,
        dataset_name: Optional[str] = None,
        *,
        schema_overrides: Optional[FieldSchemaDict] = None,
        modality: Optional[str] = None,
        id_column: Optional[str] = None,
    ) -> str:
        ds = init_dataset_source(dataset, dataset_name)
        return upload.upload_dataset(
            self._api_key,
            ds,
            schema_overrides=schema_overrides,
            modality=modality,
            id_column=id_column,
        )

    def download_cleanlab_columns(self, cleanset_id: str) -> pd.DataFrame:
        project_id = api.get_project_of_cleanset(self._api_key, cleanset_id)
        label_column = api.get_label_column_of_project(self._api_key, project_id)
        return self._download_cleanlab_columns(cleanset_id, project_id, label_column)

    def _download_cleanlab_columns(
        self, cleanset_id: str, project_id: str, label_column: str, include_action: bool = False
    ) -> pd.DataFrame:
        rows = api.download_cleanlab_columns(self._api_key, cleanset_id, all=True)
        id_col = api.get_id_column(self._api_key, cleanset_id)
        # TODO actually get _all_ the columns incl e.g., cleanlab_top_labels
        # and have the API give the column headers rather than this library hard-coding it
        headers = [
            id_col,
            "cleanlab_issue",
            "cleanlab_label_quality",
            "cleanlab_suggested_label",
            "cleanlab_clean_label",
        ]
        if include_action:
            headers.append("action")
        dataset_id = api.get_dataset_of_project(self._api_key, project_id)
        schema = api.get_dataset_schema(self._api_key, dataset_id)
        col_types = {
            id_col: as_numpy_type(schema["fields"][id_col]["data_type"]),
            "cleanlab_issue": bool,
            "cleanlab_label_quality": np.float64,
            "cleanlab_suggested_label": as_numpy_type(schema["fields"][label_column]["data_type"]),
            "cleanlab_clean_label": as_numpy_type(schema["fields"][label_column]["data_type"]),
        }
        if include_action:
            col_types["action"] = str

        # convert to dict/column-major format
        d = {
            headers[j]: np.array(
                [rows[i][j] for i in range(len(rows))], dtype=col_types[headers[j]]
            )
            for j in range(len(headers))
        }
        return pd.DataFrame(d)

    def apply_corrections(self, cleanset_id: str, dataset: Any) -> Any:
        project_id = api.get_project_of_cleanset(self._api_key, cleanset_id)
        label_column = api.get_label_column_of_project(self._api_key, project_id)
        id_col = api.get_id_column(self._api_key, cleanset_id)
        cl_cols = self._download_cleanlab_columns(
            cleanset_id, project_id, label_column, include_action=True
        )
        if pyspark_exists and isinstance(dataset, pyspark.sql.DataFrame):
            from pyspark.sql.functions import udf

            spark = dataset.sparkSession
            cl_cols_df = spark.createDataFrame(cl_cols)
            # XXX this does not handle excluded columns correctly, because the API
            # returns all rows regardless and doesn't let us distinguish between
            # excluded and non-excluded rows
            both = cl_cols_df.select([id_col, "action", "cleanlab_clean_label"]).join(
                dataset.select([id_col, label_column]),
                on=id_col,
                how="left",
            )
            final = both.withColumn(
                "__cleanlab_final_label",
                # XXX hacky, relies on no labels being "None" (the string)
                # instead, use original JSON, which uses null values where it's not specified
                udf(lambda original, clean: original if clean == "None" else clean)(
                    both[label_column],
                    "cleanlab_clean_label",
                ),
            )
            new_labels = final.select(
                [id_col, "action", "__cleanlab_final_label"]
            ).withColumnRenamed("__cleanlab_final_label", label_column)
            corrected_df = (
                dataset.drop(label_column)
                .join(new_labels, on=id_col, how="right")
                .where(new_labels["action"] != "exclude")
                .drop("action")
            )
            return corrected_df

        elif isinstance(dataset, pd.DataFrame):
            joined_ds = dataset.join(cl_cols.set_index(id_col), on=id_col)
            joined_ds["__cleanlab_final_label"] = joined_ds["cleanlab_clean_label"].where(
                joined_ds["cleanlab_clean_label"] != "None", dataset[label_column]
            )

            corrected_ds = dataset.copy()
            corrected_ds[label_column] = joined_ds["__cleanlab_final_label"]
            corrected_ds = corrected_ds[joined_ds["action"] != "exclude"]
            return corrected_ds

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
        Creates a Cleanlab Studio project

        :param dataset_id: ID of dataset to create project for
        :param project_name: name for resulting project
        :param modality: modality of project (i.e. text, tabular, image)
        :keyword task_type: type of classification to perform (i.e. multi-class, multi-label)
        :keyword model_type: type of model to train (i.e. fast, regular)
        :keyword label_column: name of column in dataset containing labels (if not supplied, we'll make our best guess)
        :keyword feature_columns: list of columns to use as features when training tabular modality project (if not supplied and modality is "tabular" we'll use all valid feature columns)
        :keyword text_column: name of column containing the text to train text modality project on (if not supplied and modality is "text" we'll make our best guess)

        :return: ID of project
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

    def get_latest_cleanset_id(self, project_id: str) -> Optional[str]:
        """
        Gets latest cleanset ID for a project when cleanset is ready

        :return: ID of latest cleanset for a project or None if an error occurred during training
        """
        return clean.get_latest_cleanset_id_when_ready(self._api_key, project_id)
