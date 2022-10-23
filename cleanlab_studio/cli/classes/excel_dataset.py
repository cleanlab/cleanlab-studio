from typing import List, Generator, Any

from cleanlab_studio.cli.classes.dataset import Dataset
import pyexcel
import pandas as pd

from cleanlab_studio.cli.types import RecordType


class ExcelDataset(Dataset):
    def read_streaming_records(self) -> Generator[RecordType, None, None]:
        for r in pyexcel.iget_records(file_name=self.filepath):
            yield r

    def read_streaming_values(self) -> Generator[List[Any], None, None]:
        stream = pyexcel.iget_array(file_name=self.filepath)
        next(stream)  # skip header row
        for r in stream:
            yield r

    def read_file_as_dataframe(self) -> pd.DataFrame:
        return pd.read_excel(self.filepath, keep_default_na=True)

    def get_columns(self) -> List[str]:
        stream = pyexcel.iget_array(file_name=self.filepath)
        return [str(col) for col in next(stream)]
