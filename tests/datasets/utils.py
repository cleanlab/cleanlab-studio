from decimal import Decimal
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from cleanlab_studio.cli.classes.csv_dataset import CsvDataset
from cleanlab_studio.cli.classes.dataset import Dataset
from cleanlab_studio.cli.classes.json_dataset import JsonDataset
from cleanlab_studio.cli.classes.excel_dataset import ExcelDataset


def csv_filepath_to_dataframe(csv_filepath: pathlib.Path) -> pd.DataFrame:
    """Reads CSV from filepath to dataframe."""
    with open(csv_filepath, **CsvDataset.READ_ARGS) as csv_file:
        return pd.read_csv(csv_file, keep_default_na=True)


def excel_filepath_to_dataframe(excel_filepath: pathlib.Path) -> pd.DataFrame:
    """Reads excel from filepath to dataframe."""
    with open(excel_filepath, **ExcelDataset.READ_ARGS) as excel_file:
        return pd.read_excel(excel_file, keep_default_na=True)


def json_filepath_to_dataframe(json_filepath: pathlib.Path) -> pd.DataFrame:
    """Reads json from filepath to dataframe."""
    with open(json_filepath, **JsonDataset.READ_ARGS) as json_file:
        return pd.read_json(json_file, orient="records", convert_axes=False, convert_dates=False)


def assert_dataset_matches_dataframe(
    dataset: Dataset, dataframe: pd.DataFrame, ignore_id: bool = False
) -> None:
    """Asserts that Cleanlab Studio Dataset object matches dataframe loaded by Pandas.

    Checks:
    - __len__
    - get_columns
    - read_streaming_records
    - read_streaming_values
    - read_file_as_dataframe
    """
    assert len(dataset) == dataframe.shape[0], "Dataset length does not match dataframe length"

    assert np.all(
        dataset.get_columns() == dataframe.columns.values
    ), "Dataset columns do not match dataframe columns"

    for dataset_row, (_, dataframe_row) in zip(
        dataset.read_streaming_records(), dataframe.iterrows()
    ):
        for dataset_val, dataframe_val in zip(dataset_row.values(), dataframe_row.values):
            assert_dataset_val_matches_dataframe_val(
                dataset_val,
                dataframe_val,
                assert_msg="Dataset read_streaming_records does not match dataframe values.",
            )

    for dataset_row, (_, dataframe_row) in zip(
        dataset.read_streaming_values(), dataframe.iterrows()
    ):
        for dataset_val, dataframe_val in zip(dataset_row, dataframe_row.values):
            assert_dataset_val_matches_dataframe_val(
                dataset_val,
                dataframe_val,
                assert_msg="Dataset read_streaming_values does not match dataframe values.",
            )

    dataset_dataframe = dataset.read_file_as_dataframe()
    if ignore_id:
        dataset_dataframe = dataset_dataframe.drop(columns=["id"])

    assert dataset_dataframe.reset_index(drop=True).equals(
        dataframe.reset_index(drop=True)
    ), "Dataset read as dataframe does not match dataframe"


def assert_dataset_val_matches_dataframe_val(
    dataset_val: Any, dataframe_val: Any, assert_msg: str
) -> None:
    """Asserts that dataset value matches dataframe value.

    Ignores nans.
    If dataset val is int and dataframe val is float, converts dataframe val to int.
    If dataset val is float and dataframe val is int, converts dataset val to int.
    If dataset val is Decimal and dataframe val is float, converts dataset val to float (and round both).
    """
    if pd.isnull(dataframe_val):
        return

    if isinstance(dataframe_val, float):
        if isinstance(dataset_val, int):
            dataframe_val = int(dataframe_val)
        elif isinstance(dataset_val, Decimal):
            dataset_val = round(float(dataset_val), ndigits=5)
            dataframe_val = round(dataframe_val, ndigits=5)

    elif isinstance(dataframe_val, int):
        if isinstance(dataset_val, float):
            dataset_val = int(dataset_val)

    assert str(dataset_val) == str(dataframe_val), assert_msg
