import os
import pathlib
from typing import Optional, TypeVar, Union

import pandas as pd
import pyspark.sql
from .dataset_source import (
    DatasetSource,
    FilepathDatasetSource,
    PandasDatasetSource,
    PySparkDatasetSource,
)


DatasetSourceType = TypeVar(
    "DatasetSourceType", bound=Union[str, os.PathLike, pd.DataFrame, pyspark.sql.DataFrame]
)


def init_dataset_source(
    dataset_source: DatasetSourceType, dataset_name: Optional[str] = None
) -> DatasetSource:
    print(type(dataset_source))
    if isinstance(dataset_source, pd.DataFrame):
        return PandasDatasetSource(df=dataset_source, dataset_name=dataset_name)
    elif isinstance(dataset_source, pyspark.sql.DataFrame):
        return PySparkDatasetSource(df=dataset_source, dataset_name=dataset_name)
    elif isinstance(dataset_source, pathlib.Path):
        return FilepathDatasetSource(filepath=dataset_source, dataset_name=dataset_name)
    elif isinstance(dataset_source, str):
        return FilepathDatasetSource(
            filepath=pathlib.Path(dataset_source), dataset_name=dataset_name
        )
    raise ValueError("Invalid dataset source provided")
