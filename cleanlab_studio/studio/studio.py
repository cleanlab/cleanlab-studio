from typing import Optional, List

import pandas as pd
import numpy as np

from cleanlab_studio.internal import api
from cleanlab_studio.internal.dataset import PandasDataset
from cleanlab_studio.internal.schema import Schema
from cleanlab_studio.internal.settings import CleanlabSettings
from cleanlab_studio.studio import upload

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
                raise ValueError("No API key found; either specify API key or log in with 'cleanlab login' first")
        api.validate_api_key(api_key)
        self._api_key = api_key


    def upload_text_dataset(
        self,
        dataset,  # for now, a spark.DataFrame; in the future, we will support more
        schema: Optional[Schema] = None,
        *,
        id: Optional[str] = None  # for resuming upload
    ) -> str:
        # actually, same as tabular
        return self.upload_tabular_dataset(dataset, schema, id=id)

    def upload_tabular_dataset(
        self,
        dataset,  # spark.DataFrame
        schema: Optional[Schema] = None,
        *,
        id: Optional[str] = None  # for resuming upload
    ) -> str:
        # TODO either schema or id must be given
        if schema is None and id is None:
            raise ValueError("Either schema or id must be provided")
        ds = PandasDataset(dataset.toPandas())
        return upload.upload_tabular_dataset(self._api_key, ds, schema, id)


    def upload_image_dataset(
        self,
        dataframe,  # spark.DataFrame
        id_column: str,
        path_column: str,
        content_column: str,
        label_column: str,
        *,
        name: str,
        id: Optional[str] = None  # for resuming upload
    ) -> str:
        return upload.upload_image_dataset(self._api_key, dataframe, name, id_column, path_column, content_column, label_column, dataset_id=id)


    def apply_corrections(self, cleanset_id: str, dataset):
        spark = dataset.sparkSession
        project_id = api.get_project_of_cleanset(self._api_key, cleanset_id)
        label_column = api.get_label_column_of_project(self._api_key, project_id)
        cl_cols = self._download_cleanlab_columns(cleanset_id, project_id, label_column)
        id_col = cl_cols.columns[0]
        cl_cols_df = spark.createDataFrame(cl_cols)
        # XXX this does not handle excluded columns correctly, because the API
        # returns all rows regardless and doesn't let us distinguish between
        # excluded and non-excluded rows
        new_labels = cl_cols_df.select([id_col, 'cleanlab_clean_label']).withColumnRenamed('cleanlab_clean_label', label_column)
        corrected_df = dataset.drop(label_column).join(new_labels, on=id_col, how='right')
        return corrected_df


    def download_cleanlab_columns(self, cleanset_id: str) -> pd.DataFrame:
        project_id = api.get_project_of_cleanset(self._api_key, cleanset_id)
        label_column = api.get_label_column_of_project(self._api_key, project_id)
        return self._download_cleanlab_columns(cleanset_id, project_id, label_column)


    def _download_cleanlab_columns(self, cleanset_id: str, project_id: str, label_column: str) -> pd.DataFrame:
        rows = api.download_cleanlab_columns(self._api_key, cleanset_id, all=True)
        id_col = api.get_id_column(self._api_key, cleanset_id)
        # TODO actually get _all_ the columns incl e.g., cleanlab_top_labels
        # and have the API give the column headers rather than this library hard-coding it
        headers = [id_col, "cleanlab_issue", "cleanlab_label_quality", "cleanlab_suggested_label", "cleanlab_clean_label"]
        dataset_id = api.get_dataset_of_project(self._api_key, project_id)
        schema = api.get_dataset_schema(self._api_key, dataset_id)
        col_types = {
            id_col: schema.fields[id_col].data_type.as_numpy_type(),
            "cleanlab_issue": bool,
            "cleanlab_label_quality": np.float64,
            "cleanlab_suggested_label": schema.fields[label_column].data_type.as_numpy_type(),
            "cleanlab_clean_label": schema.fields[label_column].data_type.as_numpy_type(),
        }

        # convert to dict/column-major format
        d = {headers[j]: np.array([rows[i][j] for i in range(len(rows))], dtype=col_types[headers[j]]) for j in range(len(headers))}
        return pd.DataFrame(d)
