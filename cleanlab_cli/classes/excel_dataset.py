from typing import List

from cleanlab_cli.classes.dataset import Dataset
import pyexcel
import pandas as pd


class ExcelDataset(Dataset):
    def read_streaming_records(self):
        for r in pyexcel.iget_records(file_name=self.filepath):
            yield r

    def read_streaming_values(self):
        stream = pyexcel.iget_array(file_name=self.filepath)
        next(stream)  # skip header row
        for r in stream:
            yield r

    def read_file_as_dataframe(self):
        return pd.read_excel(self.filepath, keep_default_na=True)

    def get_columns(self) -> List[str]:
        stream = pyexcel.iget_array(file_name=self.filepath)
        return next(stream)
