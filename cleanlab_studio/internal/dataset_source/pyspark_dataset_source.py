import json
from typing import List, IO, Iterator

from ..util import len_of_string_as_bytes, str_as_bytes

import time


try:
    import pyspark.sql
except ImportError:
    raise ImportError(
        'Must install pyspark to upload from pyspark dataframe. Use "pip install pyspark"'
    )

from .lazy_loaded_dataset_source import LazyLoadedDatasetSource


class PySparkDatasetSource(LazyLoadedDatasetSource[pyspark.sql.DataFrame]):
    def _get_size_in_bytes(self, measure_time=True) -> int:
        first = True
        size = 0

        if measure_time:
            start_time = time.time()

        for row in self.dataframe.toLocalIterator():
            if first:
                size += len_of_string_as_bytes(f"[{json.dumps(row.asDict())}")
                first = False
            else:
                size += len_of_string_as_bytes(f",{json.dumps(row.asDict())}")

        size += len_of_string_as_bytes("]")

        if measure_time:
            print(f"Time to get size: {time.time() - start_time}")

        return size

    def get_chunks(self, chunk_sizes: List[int], measure_time=True) -> Iterator[bytes]:
        first = True
        chunk = 0
        buffer = b""

        if measure_time:
            start_time = time.time()

        for row in self.dataframe.toLocalIterator():
            if first:
                buffer += str_as_bytes(f"[{json.dumps(row.asDict())}")
                first = False
            else:
                buffer += str_as_bytes(f",{json.dumps(row.asDict())}")

            if len(buffer) >= chunk_sizes[chunk]:
                yield buffer[: chunk_sizes[chunk]]
                buffer = buffer[chunk_sizes[chunk] :]
                chunk += 1

        buffer += str_as_bytes("]")

        yield buffer[: chunk_sizes[chunk]]

        if measure_time:
            print(f"Time to get chunks: {time.time() - start_time}")
