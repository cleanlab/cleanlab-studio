from typing import Any

try:
    import pyspark.sql
except ImportError:
    raise ImportError(
        'Must install pyspark to upload from pyspark dataframe. Use "pip install pyspark"'
    )

from .lazy_loaded_dataset_source import LazyLoadedDatasetSource


class PySparkDatasetSource(LazyLoadedDatasetSource[pyspark.sql.DataFrame]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
