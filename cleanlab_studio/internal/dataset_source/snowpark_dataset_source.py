from typing import List, IO, Iterator

from ..util import str_as_bytes, len_of_string_as_bytes

import time


try:
    import snowflake.snowpark as snowpark
except ImportError:
    raise ImportError(
        'Must install snowpark to upload from snowpark dataframe. Use "pip install snowflake-snowpark-python"'
    )

from .lazy_loaded_dataset_source import LazyLoadedDatasetSource


class SnowparkDatasetSource(LazyLoadedDatasetSource[snowpark.DataFrame]):
    def _get_size_in_bytes(self, measure_time=True) -> int:
        first_batch = True
        size = 0

        if measure_time:
            start_time = time.time()

        for df in self.dataframe.to_pandas_batches():
            if first_batch:
                size += len_of_string_as_bytes(f'{df.to_json(orient="records")[:-1]}')
                first_batch = False
            else:
                size += len_of_string_as_bytes(f',{df.to_json(orient="records")[1:-1]}')

        size += len_of_string_as_bytes("]")

        if measure_time:
            print(f"Time to get size: {time.time() - start_time}")

        return size

    def get_chunks(self, chunk_sizes: List[int], measure_time=True) -> Iterator[bytes]:
        first_batch = True
        chunk = 0
        buffer = b""

        if measure_time:
            start_time = time.time()

        for df in self.dataframe.to_pandas_batches():
            if first_batch:
                buffer += str_as_bytes(f'{df.to_json(orient="records")[:-1]}')
                first_batch = False
            else:
                buffer += str_as_bytes(f',{df.to_json(orient="records")[1:-1]}')

            if len(buffer) >= chunk_sizes[chunk]:
                yield buffer[: chunk_sizes[chunk]]
                buffer = buffer[chunk_sizes[chunk] :]
                chunk += 1

        buffer += str_as_bytes("]")

        yield buffer[: chunk_sizes[chunk]]

        if measure_time:
            print(f"Time to get chunks: {time.time() - start_time}")
