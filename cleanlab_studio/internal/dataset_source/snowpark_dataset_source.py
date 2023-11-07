import io
from typing import IO
import pandas as pd
import snowflake.snowpark as snowpark

from .dataframe_dataset_source import DataFrameDatasetSource


class SnowparkDatasetSource(DataFrameDatasetSource[snowpark.DataFrame]):
    def _init_fileobj_from_df(self, df: snowpark.DataFrame) -> IO[bytes]:
        fileobj = io.BytesIO()
        pd_df: pd.DataFrame = df.to_pandas()
        pd_df.to_csv(fileobj, index=False)
        fileobj.seek(0)
        return fileobj
