import pathlib
from typing import Any, Optional, TypeVar, Union, List, Dict
import math

import copy

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

# Studio team port to backend
AUTOFIX_DEFAULTS = {
    "optimized_training_data": {
        "drop_ambiguous": 0.0,
        "drop_label_issue": 0.5,
        "drop_near_duplicate": 0.2,
        "drop_outlier": 0.5,
        "relabel_confidence_threshold": 0.95,
    },
    "drop_all_issues": {
        "drop_ambiguous": 1.0,
        "drop_label_issue": 1.0,
        "drop_near_duplicate": 1.0,
        "drop_outlier": 1.0,
    },
    "suggested_actions": {
        "drop_near_duplicate": 1.0,
        "drop_outlier": 1.0,
        "relabel_confidence_threshold": 0.0,
    },
}


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

    # to use lowercase column names, they need to be wrapped in double quotes to become valid identifiers
    # for example ("col" should be '"col"' so the engine will process the name as "col")
    # https://docs.snowflake.com/en/sql-reference/identifiers-syntax
    label_col = quote(label_column)
    id_col = quote(id_col)
    action_col = quote("action")
    corrected_label_col = quote("corrected_label")
    cleanlab_final_label_col = quote("__cleanlab_final_label")

    corrected_ds = dataset
    session = dataset.session

    cl_cols_snowflake = session.create_dataframe(cl_cols)

    if id_col not in corrected_ds.columns:
        corrected_ds = corrected_ds.withColumn(id_col, monotonically_increasing_id())

    corrected_ds = (
        cl_cols_snowflake.select([id_col, action_col, corrected_label_col])
        .join(
            corrected_ds,
            on=id_col,
            how="left",
        )
        .withColumn(
            cleanlab_final_label_col,
            when(is_null(corrected_label_col), col(label_col)).otherwise(col(corrected_label_col)),
        )
        .drop(label_col, corrected_label_col)
        .withColumnRenamed(cleanlab_final_label_col, label_col)
    )

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
    from pyspark.sql.functions import row_number, monotonically_increasing_id, when, col
    from pyspark.sql.window import Window

    corrected_ds_spark = dataset.alias("corrected_ds")
    if id_col not in corrected_ds_spark.columns:
        corrected_ds_spark = corrected_ds_spark.withColumn(
            id_col,
            row_number().over(Window.orderBy(monotonically_increasing_id())) - 1,
        )
    both = cl_cols.select([id_col, "action", "corrected_label"]).join(
        corrected_ds_spark.select([id_col, label_column]),
        on=id_col,
        how="left",
    )
    final = both.withColumn(
        "__cleanlab_final_label",
        when(col("corrected_label").isNull(), col(label_column)).otherwise(col("corrected_label")),
    )
    new_labels = final.select([id_col, "action", "__cleanlab_final_label"]).withColumnRenamed(
        "__cleanlab_final_label", label_column
    )

    res = corrected_ds_spark.drop(label_column).join(new_labels, on=id_col, how="right")
    res = (
        res.where((col("action").isNull()) | (col("action") != "exclude"))
        if not keep_excluded
        else res
    ).drop("action")

    return res


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
    joined_ds["__cleanlab_final_label"] = joined_ds["corrected_label"].where(
        joined_ds["corrected_label"].notnull().tolist(),
        dataset[label_column].tolist(),
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


# Studio team port to backend
def _get_autofix_defaults_for_strategy(strategy):
    return AUTOFIX_DEFAULTS[strategy]


def _get_param_values(cleanset_df, params, strategy):
    thresholds = _get_autofix_defaults_for_strategy(strategy) if params is None else params
    param_values = {}
    for param_type, param_value in thresholds.items():
        # Convert drop fractions to number of rows and leave rest of the parameters as is
        if param_type.startswith("drop_"):
            issue_name = param_type[5:]
            num_rows = cleanset_df[f"is_{issue_name}"].sum()
            param_values[param_type] = math.ceil(num_rows * param_value)
        else:
            param_values[param_type] = param_value
    return param_values


def _get_top_fraction_ids(  # Studio team port to backend
    cleanset_df: pd.DataFrame, issue_name: str, num_rows: int, asc=True
) -> List[str]:
    """
    This will only return the IDs of datapoints to drop for a given setting of the num_rows to drop during autofix.
    Parameters:
    - cleanset_df (pd.DataFrame): The input DataFrame containing the cleanset.
    - name_col (str): The name of the column indicating the category for which the top rows should be extracted.
    - num_rows (int): The number of rows to be extracted.
    - asc (bool, optional): If True, the rows are sorted in ascending order based on the score column; if False, in descending order.
                           Default is True.

    Returns:
    - list: A list of row indices representing the top specified number of rows based on the specified score column.
    """
    bool_column_name = f"is_{issue_name}"

    # Construct a filter based on the 'label_issue' variable
    filter_condition = cleanset_df[bool_column_name]

    # Create a new DataFrame based on the filter
    filtered_df = cleanset_df[filter_condition]
    if issue_name == "near_duplicate":
        # Group by the 'near_duplicate_cluster_ID' column
        df_n = filtered_df.sort_values(by="near_duplicate_score").reset_index(drop=True)
        sorted_df = df_n.head(num_rows)
        grouped_df = sorted_df.groupby("near_duplicate_cluster_id")

        # Initialize an empty list to store the aggregated indices
        aggregated_indices = []

        # Iterate over each group
        for group_name, group_df in grouped_df:
            # Sort the group DataFrame by the 'near_duplicate_score' column in ascending order
            sorted_group_df = group_df.sort_values(
                by=f"{issue_name}_score", ascending=asc
            ).reset_index(drop=True)

            # Extract every other index and append to the aggregated indices list
            selected_indices = sorted_group_df.loc[::2, "cleanlab_row_ID"]
            aggregated_indices.extend(selected_indices)

        return aggregated_indices
    else:
        # Construct the boolean column name with 'is_' prefix and 'label_issue_score' suffix
        score_col_name = f"{issue_name}_score"

        # Sort the filtered DataFrame by the constructed boolean column in descending order
        sorted_df = filtered_df.sort_values(by=score_col_name, ascending=asc)

        # Extract the top specified number of rows and return the 'cleanlab_row_ID' column
        top_rows_ids = sorted_df["cleanlab_row_ID"].head(num_rows)

        return top_rows_ids


def _update_label_based_on_confidence(row, conf_threshold):  # Studio team port to backend
    """Update the label and is_issue based on confidence threshold if there is a label issue.

    Args:
        row (pd.Series): The row containing label information.
        conf_threshold (float): The confidence threshold for updating the label.

    Returns:
        pd.Series: The updated row.
    """
    if row["is_label_issue"] and row["suggested_label_confidence_score"] > conf_threshold:
        # make sure this does not affect back end. We are doing this to avoid dropping these datapoints in autofix later, they should be relabeled
        row["is_issue"] = False
        row["is_label_issue"] = False
        row["label"] = row["suggested_label"]
    return row


def apply_autofixed_cleanset_to_new_dataframe(  # Studio team port to backend
    original_df: pd.DataFrame, cleanset_df: pd.DataFrame, parameters: dict
) -> pd.DataFrame:
    """Apply a cleanset to update original dataaset labels and remove top rows based on specified parameters."""
    original_df_copy = copy.deepcopy(original_df)
    original_columns = original_df_copy.columns
    merged_df = pd.merge(original_df_copy, cleanset_df, left_index=True, right_on="cleanlab_row_ID")

    merged_df = merged_df.apply(
        lambda row: _update_label_based_on_confidence(
            row, conf_threshold=parameters["relabel_confidence_threshold"]
        ),
        axis=1,
    )

    indices_to_drop = _get_indices_to_drop(merged_df, parameters)

    merged_df = merged_df.drop(indices_to_drop, axis=0)
    return merged_df[original_columns]


def _get_indices_to_drop(merged_df, parameters):
    indices_to_drop = set()
    for param_name, top_num in parameters.items():
        if param_name.startswith("drop_"):
            issue_name = param_name.replace("drop_", "")
            top_percent_ids = _get_top_fraction_ids(merged_df, issue_name, top_num, asc=False)
            indices_to_drop.update(top_percent_ids)
    return list(indices_to_drop)


def quote(s: str) -> str:
    return f'"{s}"'


def quote_list(l: list) -> list:
    return [quote(i) for i in l]
