import pandas as pd
import ijson

from .dataset import Dataset


class JsonDataset(Dataset):
    def read_streaming_records(self):
        with open(self.filepath, "rb") as f:
            for r in ijson.items(f, "rows.item"):
                yield r

    def read_streaming_values(self):
        with open(self.filepath, "rb") as f:
            for r in ijson.items(f, "rows.item"):
                yield r.values()

    def read_file_as_dataframe(self):
        df = pd.read_json(self.filepath, convert_axes=False, convert_dates=False).T
        df.index = df.index.astype("str")
        df["id"] = df.index
        return df
