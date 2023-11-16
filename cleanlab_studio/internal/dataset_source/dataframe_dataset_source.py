from abc import abstractmethod
import os
import pathlib
from typing import IO, Any, Generic, TypeVar, Union

import pandas as pd

try:
    import pyspark.sql

    pyspark_exists = True
except ImportError:
    pyspark_exists = False

from .dataset_source import DatasetSource

df_type_bound = Union[pd.DataFrame, pyspark.sql.DataFrame] if pyspark_exists else pd.DataFrame

DataFrame = TypeVar("DataFrame", bound=df_type_bound)  # type: ignore


class DataFrameDatasetSource(DatasetSource, Generic[DataFrame]):
    def __init__(self, *args: Any, df: DataFrame, dataset_name: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self._fileobj = self._init_fileobj_from_df(df)
        self.file_size = self._get_size_in_bytes()
        self.file_type = "application/json"

    @abstractmethod
    def _init_fileobj_from_df(self, df: DataFrame) -> IO[bytes]:
        pass

    def _get_size_in_bytes(self) -> int:
        with self.fileobj() as dataset_file:
            dataset_file.seek(0, os.SEEK_END)
            return dataset_file.tell()

    def get_filename(self) -> str:
        return str(pathlib.Path(self.dataset_name).with_suffix(".json"))
