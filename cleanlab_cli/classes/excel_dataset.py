from typing import List

from .dataset import Dataset
import pyexcel
import pandas as pd


class ExcelDataset(Dataset):
    def read_streaming_records(self):
        for r in pyexcel.iget_records(file_name=self.filepath):
            yield r

    def read_streaming_values(self):
        for r in pyexcel.iget_records(file_name=self.filepath):
            # TODO optimize
            yield r.values()

    def read_file_as_dataframe(self):
        return pd.read_excel(self.filepath, keep_default_na=True)

    def get_columns(self) -> List[str]:
        return super().get_columns()
