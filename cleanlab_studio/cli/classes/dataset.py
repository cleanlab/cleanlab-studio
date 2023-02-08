from abc import abstractmethod
import contextlib
from typing import Optional, List, IO, Generator, Any, Iterator, Dict

import pandas as pd

from cleanlab_studio.cli.types import RecordType


class Dataset:
    READ_ARGS: Dict[str, str] = {}

    def __init__(self, filepath: Optional[str] = None, fileobj: Optional[IO] = None):
        if filepath is not None:
            self._filepath = filepath
        elif fileobj is not None:
            self._fileobj = fileobj
        else:
            raise ValueError("One of `filepath` or `fileobj` must be provided to initialize `Dataset`.")

        self._num_rows: Optional[int] = None

    @contextlib.contextmanager
    def fileobj(self) -> Iterator[IO]:
        """Yields open IO object to dataset file."""
        if self._filepath is not None:
            with open(self._filepath, **self.READ_ARGS) as dataset_file:
                yield dataset_file

        elif self._fileobj is not None:
            yield self._fileobj
            self._fileobj.seek(0)

        else:
            raise ValueError("Cannot return file object -- no filepath or fileobj available.")

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
