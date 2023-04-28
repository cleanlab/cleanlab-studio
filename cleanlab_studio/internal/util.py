import os
import pathlib
from typing import Any, Optional, TypeVar, Union

import pandas as pd
import pyspark.sql
from .dataset_source import (
    DatasetSource,
    FilepathDatasetSource,
    PandasDatasetSource,
    PySparkDatasetSource,
)


DatasetSourceType = TypeVar(
    "DatasetSourceType", bound=Union[str, os.PathLike[Any], pd.DataFrame, pyspark.sql.DataFrame]
)


def init_dataset_source(
    dataset_source: DatasetSourceType, dataset_name: Optional[str] = None
) -> DatasetSource:
    if isinstance(dataset_source, pd.DataFrame):
        assert dataset_name is not None
        return PandasDatasetSource(df=dataset_source, dataset_name=dataset_name)
    elif isinstance(dataset_source, pyspark.sql.DataFrame):
        assert dataset_name is not None
        return PySparkDatasetSource(df=dataset_source, dataset_name=dataset_name)
    elif isinstance(dataset_source, pathlib.Path):
        return FilepathDatasetSource(filepath=dataset_source, dataset_name=dataset_name)
    elif isinstance(dataset_source, str):
        return FilepathDatasetSource(
            filepath=pathlib.Path(dataset_source), dataset_name=dataset_name
        )
    raise ValueError("Invalid dataset source provided")
