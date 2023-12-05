import io
from typing import IO

import pandas as pd

try:
    import snowflake.snowpark as snowpark
except ImportError:
    raise ImportError(
        'Must install snowpark to upload from snowpark dataframe. Use "pip install snowflake-snowpark-python"'
    )

from .dataframe_dataset_source import DataFrameDatasetSource


class SnowparkDatasetSource(DataFrameDatasetSource[snowpark.DataFrame]):
    def _init_fileobj_from_df(self, df: snowpark.DataFrame) -> IO[bytes]:
        fileobj = io.BytesIO()
        pd_df: pd.DataFrame = df.to_pandas()
        pd_df.to_csv(fileobj, index=False)
        fileobj.seek(0)
        return fileobj
