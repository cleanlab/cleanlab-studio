import contextlib
import pathlib
from typing import Optional, List, IO, Generator, Iterator


class DatasetSource:
    name: str
    file_size: int
    file_type: str
    _fileobj: Optional[IO[bytes]]
    _filepath: Optional[pathlib.Path]

    def __init__(self):
        self._filepath = None
        self._fileobj = None

    @contextlib.contextmanager
    def fileobj(self) -> Iterator[IO[bytes]]:
        """Yields open IO object to dataset file."""
        if self._filepath is not None:
            with open(self._filepath, "rb") as dataset_file:  # type: ignore
                yield dataset_file

        elif self._fileobj is not None:
            try:
                yield self._fileobj
            finally:
                self._fileobj.seek(0)

        else:
            raise ValueError("Cannot return file object -- no filepath or fileobj available.")

    def get_chunks(self, chunk_sizes: List[int]) -> Iterator[bytes]:
        with self.fileobj() as f:
            return [f.read(chunk_size) for chunk_size in chunk_sizes]
