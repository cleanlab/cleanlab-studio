import io
from typing import IO
import pandas as pd

from .dataframe_dataset_source import DataFrameDatasetSource


class PandasDatasetSource(DataFrameDatasetSource[pd.DataFrame]):
    def _init_fileobj_from_df(self, df: pd.DataFrame) -> IO[bytes]:
        fileobj = io.BytesIO()
        df.to_json(fileobj, orient="records")
        fileobj.seek(0)
        return fileobj
