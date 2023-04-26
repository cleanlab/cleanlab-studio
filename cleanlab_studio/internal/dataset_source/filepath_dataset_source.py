import mimetypes
import pathlib
from typing import Any, Optional

from .dataset_source import DatasetSource


class FilepathDatasetSource(DatasetSource):
    def __init__(
        self, *args: Any, filepath: pathlib.Path, dataset_name: Optional[str] = None, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name if dataset_name is not None else filepath.name
        self.file_size = filepath.stat().st_size
        self.file_type = mimetypes.guess_type(self.name)[0]
        self._filepath = filepath

    def get_filename(self) -> str:
        return self._filepath.name
