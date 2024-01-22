from typing import Any

try:
    import snowflake.snowpark as snowpark
except ImportError:
    raise ImportError(
        'Must install snowpark to upload from snowpark dataframe. Use "pip install snowflake-snowpark-python"'
    )

from .lazy_loaded_dataset_source import LazyLoadedDatasetSource


class SnowparkDatasetSource(LazyLoadedDatasetSource[snowpark.DataFrame]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
