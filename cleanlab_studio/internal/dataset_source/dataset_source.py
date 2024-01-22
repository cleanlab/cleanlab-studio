from abc import abstractmethod
import contextlib
import pathlib
import time
from typing import Optional, List, IO, Iterator


class DatasetSource:
    dataset_name: str

    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_filename(self) -> str:
        pass
