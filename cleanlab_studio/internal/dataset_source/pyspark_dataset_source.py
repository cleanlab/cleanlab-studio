import io
from typing import IO

import pandas as pd
import pyspark.sql

from .dataframe_dataset_source import DataFrameDatasetSource


class PySparkDatasetSource(DataFrameDatasetSource[pyspark.sql.DataFrame]):
    def _init_fileobj_from_df(self, df: pyspark.sql.DataFrame) -> IO[bytes]:
        fileobj = io.BytesIO()
        pd_df: pd.DataFrame = df.toPandas()
        pd_df.to_csv(fileobj)
        fileobj.seek(0)
        return fileobj
