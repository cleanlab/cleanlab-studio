from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import pyspark.sql

    pyspark_exists = True
except ImportError:
    pyspark_exists = False

from . import upload
from cleanlab_studio.internal.api import api
from cleanlab_studio.internal.util import init_dataset_source, as_numpy_type
from cleanlab_studio.internal.settings import CleanlabSettings
from cleanlab_studio.internal.types import FieldSchemaDict


class Studio:
    _api_key: str

    def __init__(self, api_key: Optional[str]):
        api.check_client_version()
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
        self, cleanset_id: str, project_id: str, label_column: str
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
        dataset_id = api.get_dataset_of_project(self._api_key, project_id)
        schema = api.get_dataset_schema(self._api_key, dataset_id)
        col_types = {
            id_col: as_numpy_type(schema["fields"][id_col]["data_type"]),
            "cleanlab_issue": bool,
            "cleanlab_label_quality": np.float64,
            "cleanlab_suggested_label": as_numpy_type(schema["fields"][label_column]["data_type"]),
            "cleanlab_clean_label": as_numpy_type(schema["fields"][label_column]["data_type"]),
        }

        # convert to dict/column-major format
        d = {
            headers[j]: np.array(
                [rows[i][j] for i in range(len(rows))], dtype=col_types[headers[j]]
            )
            for j in range(len(headers))
        }
        return pd.DataFrame(d)

    def apply_corrections(self, cleanset_id: str, dataset: Any) -> Any:
        if pyspark_exists and isinstance(dataset, pyspark.sql.DataFrame):
            from pyspark.sql.functions import udf

            spark = dataset.sparkSession
            project_id = api.get_project_of_cleanset(self._api_key, cleanset_id)
            label_column = api.get_label_column_of_project(self._api_key, project_id)
            id_col = api.get_id_column(self._api_key, cleanset_id)
            cl_cols = self._download_cleanlab_columns(cleanset_id, project_id, label_column)
            cl_cols_df = spark.createDataFrame(cl_cols)
            # XXX this does not handle excluded columns correctly, because the API
            # returns all rows regardless and doesn't let us distinguish between
            # excluded and non-excluded rows
            both = cl_cols_df.select([id_col, "cleanlab_clean_label"]).join(
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
            new_labels = final.select([id_col, "__cleanlab_final_label"]).withColumnRenamed(
                "__cleanlab_final_label", label_column
            )
            corrected_df = dataset.drop(label_column).join(new_labels, on=id_col, how="right")
            return corrected_df

        elif isinstance(dataset, pd.DataFrame):
            project_id = api.get_project_of_cleanset(self._api_key, cleanset_id)
            label_column = api.get_label_column_of_project(self._api_key, project_id)
            id_col = api.get_id_column(self._api_key, cleanset_id)
            cl_cols = self._download_cleanlab_columns(cleanset_id, project_id, label_column)

            joined_ds = dataset.join(cl_cols.set_index(id_col), on=id_col)
            joined_ds["__cleanlab_final_label"] = joined_ds["cleanlab_clean_label"].where(
                joined_ds["cleanlab_clean_label"] != "None", dataset[label_column]
            )

            corrected_ds = dataset.copy()
            corrected_ds[label_column] = joined_ds["__cleanlab_final_label"]
            return corrected_ds
