from typing import Generator, List, Any

import pandas as pd

from .dataset import Dataset
from ..types import RecordType

class PandasDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df

    def read_streaming_records(self) -> Generator[RecordType, None, None]:
        for idx, row in self.df.iterrows():
            yield {str(k): v for k, v in row.to_dict().items()}

    def read_streaming_values(self) -> Generator[List[Any], None, None]:
        for idx, row in self.df.iterrows():
            yield row.tolist()

    def read_file_as_dataframe(self) -> pd.DataFrame:
        return self.df
