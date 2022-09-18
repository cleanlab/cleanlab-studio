import csv
from typing import List, Generator, Any

import pandas as pd

from .dataset import Dataset
from ..types import RecordType


class CsvDataset(Dataset):
    def count_rows(self) -> int:
        with open(self.filepath, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            return len(list(reader)) - 1  # first row is headers

    def get_columns(self) -> List[str]:
        with open(self.filepath, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            columns = next(reader)
            return columns

    def read_streaming_records(self) -> Generator[RecordType, None, None]:
        with open(self.filepath, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            columns = next(reader)
            for row in reader:
                yield dict(zip(columns, row))

    def read_streaming_values(self) -> Generator[List[Any], None, None]:
        with open(self.filepath, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                yield row

    def read_file_as_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self.filepath, keep_default_na=True)
