from abc import abstractmethod
from typing import Optional, List, Dict, Generator, Any

import pandas as pd

from cleanlab_studio.cli.types import RecordType


class Dataset:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._num_rows: Optional[int] = None

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
