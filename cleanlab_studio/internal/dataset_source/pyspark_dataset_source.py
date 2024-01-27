from typing import Any

try:
    import pyspark.sql
except ImportError:
    raise ImportError(
        'Must install pyspark to upload from pyspark dataframe. Use "pip install pyspark"'
    )

from pyspark.sql import DataFrame
from .lazy_loaded_dataset_source import LazyLoadedDatasetSource


class PySparkDatasetSource(LazyLoadedDatasetSource[pyspark.sql.DataFrame]):
    def __init__(self, df: DataFrame, *args: Any, **kwargs: Any) -> None:
        self.dataframe = df
        super().__init__(*args, **kwargs)

    def _get_rows(self) -> int:
        return self.dataframe.count()

    def get_rows_iterator(self) -> Any:
        return self.dataframe.toLocalIterator()
