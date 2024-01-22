import io
import os
import pathlib
from typing import IO, Any
import pandas as pd

from .local_dataset_source import LocalDatasetSource


class PandasDatasetSource(LocalDatasetSource):
    def __init__(self, *args: Any, df: pd.DataFrame, dataset_name: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self._fileobj = self._init_fileobj_from_df(df)
        self.file_size = self._get_size_in_bytes()
        self.file_type = "application/json"

    def _init_fileobj_from_df(self, df: pd.DataFrame) -> IO[bytes]:
        fileobj = io.BytesIO()
        df.to_json(fileobj, orient="records")
        fileobj.seek(0)
        return fileobj

    def _get_size_in_bytes(self) -> int:
        with self.fileobj() as dataset_file:
            dataset_file.seek(0, os.SEEK_END)
            return dataset_file.tell()

    def get_filename(self) -> str:
        return str(pathlib.Path(self.dataset_name).with_suffix(".json"))
