import pathlib
from typing import Any, Optional, TypeVar, Union
import math

import numpy as np
import pandas as pd

try:
    import snowflake.snowpark as snowpark

    snowpark_exists = True
except ImportError:
    snowpark_exists = False

try:
    import pyspark.sql

    pyspark_exists = True
except ImportError:
    pyspark_exists = False

from .dataset_source import (
    DatasetSource,
    FilepathDatasetSource,
    PandasDatasetSource,
)

dataset_source_types = [str, pathlib.Path, pd.DataFrame]
if pyspark_exists:
    dataset_source_types.append(pyspark.sql.DataFrame)
if snowpark_exists:
    dataset_source_types.append(snowpark.DataFrame)

DatasetSourceType = TypeVar("DatasetSourceType", bound=Union[tuple(dataset_source_types)])  # type: ignore


def init_dataset_source(
    dataset_source: DatasetSourceType, dataset_name: Optional[str] = None
) -> DatasetSource:
    if isinstance(dataset_source, pd.DataFrame):
        if dataset_name is None:
            raise ValueError("Must provide dataset name if uploading from a DataFrame")
        return PandasDatasetSource(df=dataset_source, dataset_name=dataset_name)
    elif isinstance(dataset_source, pathlib.Path):
        return FilepathDatasetSource(filepath=dataset_source, dataset_name=dataset_name)
    elif isinstance(dataset_source, str):
        return FilepathDatasetSource(
            filepath=pathlib.Path(dataset_source), dataset_name=dataset_name
        )
    elif snowpark_exists and isinstance(dataset_source, snowpark.DataFrame):
        from .dataset_source import SnowparkDatasetSource

        if dataset_name is None:
            raise ValueError("Must provide dataset name if uploading from a DataFrame")
        return SnowparkDatasetSource(df=dataset_source, dataset_name=dataset_name)
    elif pyspark_exists and isinstance(dataset_source, pyspark.sql.DataFrame):
        from .dataset_source import PySparkDatasetSource

        if dataset_name is None:
            raise ValueError("Must provide dataset name if uploading from a DataFrame")
        return PySparkDatasetSource(df=dataset_source, dataset_name=dataset_name)
    else:
        raise ValueError("Invalid dataset source provided")


def apply_corrections_snowpark_df(
    dataset: Any,
    cl_cols: Any,
    id_col: str,
    label_column: str,
    keep_excluded: bool,
) -> Any:
    from snowflake.snowpark.functions import (
        when,
        col,
        is_null,
        monotonically_increasing_id,
    )

    label_col = quote(label_column)
    id_col = quote(id_col)
    action_col = quote("action")
    corrected_label_col = quote("corrected_label")
    cleanlab_final_label_col = quote("__cleanlab_final_label")

    corrected_ds = dataset
    session = dataset.session

    cl_cols = session.create_dataframe(cl_cols)

    if id_col not in corrected_ds.columns:
        corrected_ds = corrected_ds.withColumn(id_col, monotonically_increasing_id())

    both = cl_cols.select([id_col, action_col, corrected_label_col]).join(
        corrected_ds.select([id_col, label_col]),
        on=id_col,
        how="left",
    )

    final = both.withColumn(
        cleanlab_final_label_col,
        when(is_null(corrected_label_col), col(label_col)).otherwise(col(corrected_label_col)),
    )

    new_labels = final.select([id_col, action_col, cleanlab_final_label_col]).withColumnRenamed(
        cleanlab_final_label_col, label_col
    )

    corrected_ds = corrected_ds.drop(label_col).join(new_labels, on=id_col, how="left")
    corrected_ds = (
        corrected_ds.where((col(action_col) != "exclude") | is_null(col(action_col)))
        if not keep_excluded
        else corrected_ds
    ).drop(action_col)

    return corrected_ds


def apply_corrections_spark_df(
    dataset: Any,
    cl_cols: Any,
    id_col: str,
    label_column: str,
    keep_excluded: bool,
) -> Any:
    from pyspark.sql.functions import udf

    corrected_ds_spark = dataset.alias("corrected_ds")
    if id_col not in corrected_ds_spark.columns:
        from pyspark.sql.functions import (
            row_number,
            monotonically_increasing_id,
        )
        from pyspark.sql.window import Window

        corrected_ds_spark = corrected_ds_spark.withColumn(
            id_col,
            row_number().over(Window.orderBy(monotonically_increasing_id())) - 1,
        )
    both = cl_cols.select([id_col, "action", "clean_label"]).join(
        corrected_ds_spark.select([id_col, label_column]),
        on=id_col,
        how="left",
    )
    final = both.withColumn(
        "__cleanlab_final_label",
        # XXX hacky, checks if label is none by hand
        # instead, use original JSON, which uses null values where it's not specified
        udf(lambda original, clean: original if check_none(clean) else clean)(
            both[label_column],
            "clean_label",
        ),
    )
    new_labels = final.select([id_col, "action", "__cleanlab_final_label"]).withColumnRenamed(
        "__cleanlab_final_label", label_column
    )
    return (
        corrected_ds_spark.drop(label_column)
        .join(new_labels, on=id_col, how="right")
        .where(new_labels["action"] != "exclude")
        .drop("action")
    )


def apply_corrections_pd_df(
    dataset: pd.DataFrame,
    cl_cols: pd.DataFrame,
    id_col: str,
    label_column: str,
    keep_excluded: bool,
) -> pd.DataFrame:
    joined_ds: pd.DataFrame
    if id_col in dataset.columns:
        joined_ds = dataset.join(cl_cols.set_index(id_col), on=id_col)
    else:
        joined_ds = dataset.join(cl_cols.set_index(id_col).sort_values(by=id_col))
    joined_ds["__cleanlab_final_label"] = joined_ds["clean_label"].where(
        np.asarray(list(map(check_not_none, joined_ds["clean_label"].to_numpy()))),
        dataset[label_column].to_numpy(),
    )

    corrected_ds: pd.DataFrame = dataset.copy()
    corrected_ds[label_column] = joined_ds["__cleanlab_final_label"]
    if not keep_excluded:
        corrected_ds = corrected_ds.loc[(joined_ds["action"] != "exclude").fillna(True)]
    else:
        corrected_ds["action"] = joined_ds["action"]
    return corrected_ds


def check_none(x: Any) -> bool:
    if isinstance(x, str):
        return x == "None" or x == "none" or x == "null" or x == "NULL"
    elif isinstance(x, float):
        return math.isnan(x)
    elif pd.isnull(x):
        return True
    else:
        return x is None


def check_not_none(x: Any) -> bool:
    return not check_none(x)


def quote(s: str) -> str:
    return f'"{s}"'


def quote_list(l: list) -> list:
    return list(map(quote, l))
