import mimetypes
import pathlib
from typing import Any

from .dataset_source import DatasetSource


class FilepathDatasetSource(DatasetSource):
    def __init__(self, *args: Any, filepath: pathlib.Path, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.name = filepath.name
        self.file_size = filepath.stat().st_size
        self.file_type = mimetypes.guess_type(self.name)[0]
        self._filepath = filepath
