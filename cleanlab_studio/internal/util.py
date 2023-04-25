import pathlib
from typing import Any, Optional

import pandas as pd
from .dataset_source import DatasetSource, FilepathDatasetSource, PandasDatasetSource


def init_dataset_source(dataset_source: Any, name: Optional[str] = None) -> DatasetSource:
    if isinstance(dataset_source, pd.DataFrame):
        return PandasDatasetSource(df=dataset_source, name=name)
    elif isinstance(dataset_source, pathlib.Path):
        return FilepathDatasetSource(filepath=dataset_source, name=name)
    elif isinstance(dataset_source, str):
        return FilepathDatasetSource(filepath=pathlib.Path(dataset_source), name=name)
    raise ValueError("Invalid dataset source provided")
