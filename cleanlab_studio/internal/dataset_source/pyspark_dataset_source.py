import io
from typing import IO

import pandas as pd

try:
    import pyspark.sql
except ImportError:
    raise ImportError(
        'Must install pyspark to upload from pyspark dataframe. Use "pip install pyspark"'
    )

from .dataframe_dataset_source import DataFrameDatasetSource


class PySparkDatasetSource(DataFrameDatasetSource[pyspark.sql.DataFrame]):
    def _init_fileobj_from_df(self, df: pyspark.sql.DataFrame) -> IO[bytes]:
        fileobj = io.BytesIO()
        pd_df: pd.DataFrame = df.toPandas()
        pd_df.to_json(fileobj, orient="records")
        fileobj.seek(0)
        return fileobj
