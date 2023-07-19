import mimetypes
import pathlib
from typing import Any, Optional

from .dataset_source import DatasetSource


class FilepathDatasetSource(DatasetSource):
    def __init__(
        self,
        *args: Any,
        filepath: pathlib.Path,
        dataset_name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name if dataset_name is not None else filepath.name
        self.file_size = filepath.stat().st_size
        maybe_file_type = mimetypes.guess_type(filepath)[0]
        if maybe_file_type is None:
            raise ValueError(
                f"Could not identify type of file at {filepath}. Make sure file name has valid extension"
            )
        self.file_type = maybe_file_type
        self._filepath = filepath

    def get_filename(self) -> str:
        assert self._filepath is not None
        return self._filepath.name
