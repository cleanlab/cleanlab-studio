from abc import abstractmethod
import contextlib
from typing import Optional, List, IO, Generator, Any, Iterator, Dict, Generic, TypeVar, Union

import pandas as pd

from cleanlab_studio.cli.types import RecordType
from cleanlab_studio.errors import InvalidDatasetError, MissingPathError


FileObj = TypeVar("FileObj", bound=Union[IO[str], IO[bytes]])


class Dataset(Generic[FileObj]):
    READ_ARGS: Dict[str, str] = {}

    def __init__(self, filepath: Optional[str] = None, fileobj: Optional[FileObj] = None):
        if filepath is None and fileobj is None:
            raise InvalidDatasetError(
                "One of `filepath` or `fileobj` must be provided to initialize `Dataset`."
            )

        self._filepath: Optional[str] = filepath
        self._fileobj: Optional[FileObj] = fileobj

        self._num_rows: Optional[int] = None

    @contextlib.contextmanager
    def fileobj(self) -> Iterator[FileObj]:
        """Yields open IO object to dataset file."""
        if self._filepath is not None:
            with open(self._filepath, **self.READ_ARGS) as dataset_file:  # type: ignore
                yield dataset_file

        elif self._fileobj is not None:
            try:
                yield self._fileobj
            finally:
                self._fileobj.seek(0)

        else:
            raise MissingPathError("Cannot return file object -- no filepath or fileobj available.")

    def __len__(self) -> int:
        if self._num_rows is None:
            self._num_rows = self.count_rows()
        return self._num_rows

    def count_rows(self) -> int:
        count = 0
        for _ in self.read_streaming_values():
            count += 1
        return count

    def get_columns(self) -> List[str]:
        stream = self.read_streaming_records()
        return list(next(stream).keys())

    @abstractmethod
    def read_streaming_records(self) -> Generator[RecordType, None, None]:
        pass

    @abstractmethod
    def read_streaming_values(self) -> Generator[List[Any], None, None]:
        pass

    @abstractmethod
    def read_file_as_dataframe(self) -> pd.DataFrame:
        pass
