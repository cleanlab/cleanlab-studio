from typing import Any

try:
    import snowflake.snowpark as snowpark
except ImportError:
    raise ImportError(
        'Must install snowpark to upload from snowpark dataframe. Use "pip install snowflake-snowpark-python"'
    )

from snowflake.snowpark import DataFrame
from .lazy_loaded_dataset_source import LazyLoadedDatasetSource


class SnowparkDatasetSource(LazyLoadedDatasetSource[snowpark.DataFrame]):
    def __init__(self, df: DataFrame, *args: Any, **kwargs: Any) -> None:
        self.dataframe = df
        super().__init__(*args, **kwargs)

    def _get_rows(self) -> int:
        return self.dataframe.count()

    def get_rows_iterator(self) -> Any:
        return self.dataframe.toLocalIterator()
