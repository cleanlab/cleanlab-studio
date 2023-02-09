from typing import Generator, List, Any, Dict, IO

import pandas as pd
import ijson

from .dataset import Dataset
from cleanlab_studio.cli.types import RecordType


class JsonDataset(Dataset[IO[bytes]]):
    READ_ARGS: Dict[str, str] = {"mode": "rb"}

    def read_streaming_records(self) -> Generator[RecordType, None, None]:
        with self.fileobj() as f:
            for r in ijson.items(f, "item"):
                yield r

    def read_streaming_values(self) -> Generator[List[Any], None, None]:
        with self.fileobj() as f:
            for r in ijson.items(f, "item"):
                yield r.values()

    def read_file_as_dataframe(self) -> pd.DataFrame:
        with self.fileobj() as f:
            df = pd.read_json(f, orient="records", convert_axes=False, convert_dates=False)

        df.index = df.index.astype("str")
        df["id"] = df.index
        return df
