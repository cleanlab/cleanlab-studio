import csv
from typing import List, Generator, Any, Dict, IO

import pandas as pd

from .dataset import Dataset
from ..types import RecordType


class CsvDataset(Dataset[IO[str]]):
    READ_ARGS: Dict[str, Any] = {
        "mode": "r",
        "encoding": "utf-8",
        "errors": "ignore",
    }

    def count_rows(self) -> int:
        with self.fileobj() as f:
            reader = csv.reader(f)

            # handle case where CSV is empty
            return max(
                len(list(reader)) - 1,  # first row is headers
                0,
            )

    def get_columns(self) -> List[str]:
        with self.fileobj() as f:
            reader = csv.reader(f)
            columns = next(reader)
            return columns

    def read_streaming_records(self) -> Generator[RecordType, None, None]:
        with self.fileobj() as f:
            reader = csv.reader(f)
            columns = next(reader)
            for row in reader:
                yield dict(zip(columns, row))

    def read_streaming_values(self) -> Generator[List[Any], None, None]:
        with self.fileobj() as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                yield row

    def read_file_as_dataframe(self) -> pd.DataFrame:
        with self.fileobj() as f:
            return pd.read_csv(f, keep_default_na=True)
