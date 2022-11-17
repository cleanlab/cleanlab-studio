from typing import Generator, List, Any

import pandas as pd
import ijson

from .dataset import Dataset
from cleanlab_studio.cli.types import RecordType


class JsonDataset(Dataset):
    def read_streaming_records(self) -> Generator[RecordType, None, None]:
        with open(self.filepath, "rb") as f:
            for r in ijson.items(f, "item"):
                yield r

    def read_streaming_values(self) -> Generator[List[Any], None, None]:
        with open(self.filepath, "rb") as f:
            for r in ijson.items(f, "item"):
                yield r.values()

    def read_file_as_dataframe(self) -> pd.DataFrame:
        df = pd.read_json(self.filepath, convert_axes=False, convert_dates=False).T
        df.index = df.index.astype("str")
        df["id"] = df.index
        return df
