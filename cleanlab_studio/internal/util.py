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
    "DatasetSourceType", bound=Union[str, pathlib.Path, pd.DataFrame, pyspark.sql.DataFrame]
)


def init_dataset_source(
    dataset_source: DatasetSourceType, dataset_name: Optional[str] = None
) -> DatasetSource:
    if isinstance(dataset_source, pd.DataFrame):
        if dataset_name is None:
            raise ValueError("Must provide dataset name if uploading from a DataFrame")
        return PandasDatasetSource(df=dataset_source, dataset_name=dataset_name)
    elif isinstance(dataset_source, pyspark.sql.DataFrame):
        if dataset_name is None:
            raise ValueError("Must provide dataset name if uploading from a DataFrame")
        return PySparkDatasetSource(df=dataset_source, dataset_name=dataset_name)
    elif isinstance(dataset_source, pathlib.Path):
        return FilepathDatasetSource(filepath=dataset_source, dataset_name=dataset_name)
    elif isinstance(dataset_source, str):
        return FilepathDatasetSource(
            filepath=pathlib.Path(dataset_source), dataset_name=dataset_name
        )
    raise ValueError("Invalid dataset source provided")
