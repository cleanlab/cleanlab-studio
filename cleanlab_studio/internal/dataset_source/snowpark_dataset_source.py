import io
from typing import IO

import pandas as pd

from cleanlab_studio.errors import NotInstalledError

try:
    import snowflake.snowpark as snowpark
except ImportError:
    raise NotInstalledError(
        'Must install snowpark to upload from snowpark dataframe. Use "pip install snowflake-snowpark-python"'
    )

from .dataframe_dataset_source import DataFrameDatasetSource


class SnowparkDatasetSource(DataFrameDatasetSource[snowpark.DataFrame]):
    def _init_fileobj_from_df(self, df: snowpark.DataFrame) -> IO[bytes]:
        fileobj = io.BytesIO()
        pd_df: pd.DataFrame = df.to_pandas()
        pd_df.to_json(fileobj, index=False, orient="records")
        fileobj.seek(0)
        return fileobj
