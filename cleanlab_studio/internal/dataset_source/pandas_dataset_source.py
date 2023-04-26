import io
import os
import pathlib
import tempfile
from typing import IO, Any
import pandas as pd

from .dataset_source import DatasetSource


class PandasDatasetSource(DatasetSource):
    def __init__(self, *args: Any, df: pd.DataFrame, name: str, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.name = (
            name if pathlib.Path(name).suffix else str(pathlib.Path(name).with_suffix(".csv"))
        )
        self._fileobj = self._init_fileobj_from_df(df)
        self.file_size = self._get_size_in_bytes()
        self.file_type = "text/csv"

    def _init_fileobj_from_df(self, df: pd.DataFrame) -> IO[bytes]:
        fileobj = tempfile.NamedTemporaryFile(delete=False)
        df.to_csv(fileobj)
        fileobj.seek(0)
        return fileobj

    def _get_size_in_bytes(self) -> int:
        with self.fileobj() as dataset_file:
            dataset_file.seek(0, os.SEEK_END)
            return dataset_file.tell()
