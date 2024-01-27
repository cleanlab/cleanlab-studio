from .dataset_source import DatasetSource as DatasetSource
from .local_dataset_source import LocalDatasetSource as LocalDatasetSource
from .filepath_dataset_source import FilepathDatasetSource as FilepathDatasetSource
from .pandas_dataset_source import PandasDatasetSource as PandasDatasetSource


snowflake_exists = False
pyspark_exists = False

try:
    import snowflake.snowpark

    snowflake_exists = True
    from .snowpark_dataset_source import SnowparkDatasetSource as SnowparkDatasetSource
except ImportError:
    pass

try:
    import pyspark.sql

    pyspark_exists = True
    from .pyspark_dataset_source import PySparkDatasetSource as PySparkDatasetSource
except ImportError:
    pass

if snowflake_exists or pyspark_exists:
    from .lazy_loaded_dataset_source import LazyLoadedDatasetSource as LazyLoadedDatasetSource
