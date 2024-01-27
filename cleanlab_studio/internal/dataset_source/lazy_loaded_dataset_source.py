from abc import abstractmethod
import pathlib
import json

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
    def __init__(self, *args: Any, dataset_name: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.total_rows = self._get_rows()
        self.file_type = "application/json"

    @abstractmethod
    def _get_rows(self) -> int:
        pass

    @abstractmethod
    def get_rows_iterator(self) -> Any:
        pass

    def get_chunks(self, chunk_size: int) -> Iterator[tuple[str, int]]:
        first = True
        chunk = 0
        rows = 0
        buffer = ""

        for row in self.get_rows_iterator():
            if first:
                buffer += f"[{json.dumps(row.asDict())}"
                first = False
            else:
                buffer += f",{json.dumps(row.asDict())}"

            if len(buffer) >= chunk_size:
                yield buffer[:chunk_size], rows
                buffer = buffer[chunk_size:]
                chunk += 1
            rows += 1

        buffer += "]"

        yield buffer[:chunk_size], rows

    @abstractmethod
    def _get_size_in_bytes(self) -> int:
        pass

    def get_filename(self) -> str:
        return str(pathlib.Path(self.dataset_name).with_suffix(".json"))

    def get_file_type(self) -> str:
        return self.file_type
