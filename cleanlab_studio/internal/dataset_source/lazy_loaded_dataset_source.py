from abc import abstractmethod
import pathlib
from typing import IO, Any, Generic, Iterator, TypeVar, Union

try:
    import pyspark.sql

    pyspark_exists = True
except ImportError:
    pyspark_exists = False

try:
    import snowflake.snowpark

    snowpark_exists = True
except ImportError:
    snowpark_exists = False


from .dataset_source import DatasetSource

df_type_bound = []
if pyspark_exists:
    df_type_bound.append(pyspark.sql.DataFrame)
if snowpark_exists:
    df_type_bound.append(snowflake.snowpark.DataFrame)

df_type_bound = Union[tuple(df_type_bound)] if len(df_type_bound) >= 2 else df_type_bound[0]

DataFrame = TypeVar("DataFrame", bound=df_type_bound)  # type: ignore


class LazyLoadedDatasetSource(DatasetSource, Generic[DataFrame]):
    def __init__(self, *args: Any, df: DataFrame, dataset_name: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dataframe = df
        self.dataset_name = dataset_name
        self.file_size = self._get_size_in_bytes()
        self.file_type = "application/json"

    def fileobj(self) -> Iterator[IO[bytes]]:
        # lazy loaded dataframes might be too large to fit in memory or disk
        # so this class does not create a file object
        yield None

    @abstractmethod
    def _get_size_in_bytes(self) -> int:
        pass

    def get_filename(self) -> str:
        return str(pathlib.Path(self.dataset_name).with_suffix(".json"))
