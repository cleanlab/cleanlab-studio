from abc import abstractmethod
import pathlib
import json
from ..util import str_as_bytes

from typing import Any, Generic, Iterator, TypeVar, Union

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
        self.total_rows = self._get_rows()
        self.file_type = "application/json"

    def _get_rows(self) -> int:
        return self.dataframe.count()

    def get_chunks(self, chunk_size: int) -> Iterator[(bytes, int)]:
        first = True
        chunk = 0
        rows = 0
        buffer = b""

        for row in self.dataframe.toLocalIterator():
            if first:
                buffer += str_as_bytes(f"[{json.dumps(row.asDict())}")
                first = False
            else:
                buffer += str_as_bytes(f",{json.dumps(row.asDict())}")

            if len(buffer) >= chunk_size:
                yield buffer[:chunk_size], rows
                buffer = buffer[chunk_size:]
                chunk += 1
            rows += 1

        buffer += str_as_bytes("]")

        yield buffer[:chunk_size], rows

    @abstractmethod
    def _get_size_in_bytes(self) -> int:
        pass

    def get_filename(self) -> str:
        return str(pathlib.Path(self.dataset_name).with_suffix(".json"))
