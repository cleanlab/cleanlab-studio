import pathlib
from typing import Any, Optional, TypeVar, Union
import math

import copy

import pandas as pd

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

dataset_source_types = (
    Union[str, pathlib.Path, pd.DataFrame]
    if not pyspark_exists
    else Union[str, pathlib.Path, pd.DataFrame, pyspark.sql.DataFrame]
)

DatasetSourceType = TypeVar("DatasetSourceType", bound=dataset_source_types)  # type: ignore


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
    elif pyspark_exists and isinstance(dataset_source, pyspark.sql.DataFrame):
        from .dataset_source import PySparkDatasetSource

        if dataset_name is None:
            raise ValueError("Must provide dataset name if uploading from a DataFrame")
        return PySparkDatasetSource(df=dataset_source, dataset_name=dataset_name)
    else:
        raise ValueError("Invalid dataset source provided")


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


def _get_autofix_default_params():
    """returns default params of autofix"""
    return {
        "ambiguous": 0.2,
        "label_issue": 0.5,
        "near_duplicate": 0.2,
        "outlier": 0.5,
        "confidence_threshold": 0.95,
    }


def _get_autofix_defaults(cleanset_df):
    """
    Generate default values for autofix parameters based on the size of the cleaned dataset.
    """
    default_params = _get_autofix_default_params()
    default_values = {}

    for param_name, param_value in default_params.items():
        if param_name != "confidence_threshold":
            num_rows = cleanset_df[f"is_{param_name}"].sum()
            default_values[param_name] = math.ceil(num_rows * param_value)
        else:
            default_values[param_name] = param_value
    return default_values


def _get_top_fraction_ids(cleanset_df, name_col, num_rows, asc=True):
    """
    Extracts the top specified number of rows based on a specified score column from a DataFrame.

    Parameters:
    - cleanset_df (pd.DataFrame): The input DataFrame containing the cleanset.
    - name_col (str): The name of the column indicating the category for which the top rows should be extracted.
    - num_rows (int): The number of rows to be extracted.
    - asc (bool, optional): If True, the rows are sorted in ascending order based on the score column; if False, in descending order.
                           Default is True.

    Returns:
    - list: A list of row indices representing the top specified number of rows based on the specified score column.
    """
    bool_column_name = f"is_{name_col}"

    # Construct a filter based on the 'label_issue' variable
    filter_condition = cleanset_df[bool_column_name]

    # Create a new DataFrame based on the filter
    filtered_df = cleanset_df[filter_condition]
    if name_col == "near_duplicate":
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
                by=f"{name_col}_score", ascending=asc
            ).reset_index(drop=True)

            # Extract every other index and append to the aggregated indices list
            selected_indices = sorted_group_df.loc[::2, "cleanlab_row_ID"]
            aggregated_indices.extend(selected_indices)

        return aggregated_indices
    else:
        # Construct the boolean column name with 'is_' prefix and 'label_issue_score' suffix
        score_col_name = f"{name_col}_score"

        # Sort the filtered DataFrame by the constructed boolean column in descending order
        sorted_df = filtered_df.sort_values(by=score_col_name, ascending=asc)

        # Extract the top specified number of rows and return the 'cleanlab_row_ID' column
        top_rows_ids = sorted_df["cleanlab_row_ID"].head(num_rows)

        return top_rows_ids


def _update_label_based_on_confidence(row, conf_threshold):
    """Update the label and is_issue based on confidence threshold if there is a label issue.

    Args:
        row (pd.Series): The row containing label information.
        conf_threshold (float): The confidence threshold for updating the label.

    Returns:
        pd.Series: The updated row.
    """
    if row["is_label_issue"] and row["suggested_label_confidence_score"] > conf_threshold:
        row["is_issue"] = False
        row["label"] = row["suggested_label"]
    return row


def _apply_autofixed_cleanset_to_new_dataframe(original_df, cleanset_df, parameters):
    """Apply a cleanset to update original dataaset labels and remove top rows based on specified parameters."""
    original_df_copy = copy.deepcopy(original_df)
    original_columns = original_df_copy.columns
    merged_df = pd.merge(original_df_copy, cleanset_df, left_index=True, right_on="cleanlab_row_ID")

    merged_df = merged_df.apply(
        lambda row: _update_label_based_on_confidence(
            row, conf_threshold=parameters["confidence_threshold"]
        ),
        axis=1,
    )

    indices_to_drop = set()
    for column_name, top_num in parameters.items():
        if column_name == "confidence_threshold":
            continue
        top_percent_ids = _get_top_fraction_ids(merged_df, column_name, top_num, asc=False)
        indices_to_drop.update(top_percent_ids)

    merged_df = merged_df.drop(list(indices_to_drop), axis=0).reset_index(drop=True)
    return merged_df[original_columns]
