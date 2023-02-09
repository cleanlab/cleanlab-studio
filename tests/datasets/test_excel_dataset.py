import pathlib
from typing import List

import pandas as pd
import pytest

from cleanlab_studio.cli.classes.excel_dataset import ExcelDataset
from .constants import DATASETS_DIR
from .utils import assert_dataset_matches_dataframe, excel_filepath_to_dataframe


EXCEL_DATASETS: List[pathlib.Path] = [*DATASETS_DIR.glob(r"*.xlsx")]


@pytest.mark.parametrize(
    ("excel_filepath", "excel_df"),
    (
        (excel_filepath, excel_filepath_to_dataframe(excel_filepath))
        for excel_filepath in EXCEL_DATASETS
    ),
)
def test_excel_dataset_from_filepath_matches_expected(
    excel_filepath: pathlib.Path, excel_df: pd.DataFrame
):
    """Tests that ExcelDataset, loaded from filepath, matches excel loaded by Pandas.

    Checks:
    - __len__
    - get_columns
    - read_streaming_records
    - read_streaming_values
    - read_file_as_dataframe
    """
    # load ExcelDataset from filepath
    excel_dataset = ExcelDataset(filepath=str(excel_filepath), file_type="xlsx")

    # check that methods match values read by pandas
    assert_dataset_matches_dataframe(excel_dataset, excel_df)


@pytest.mark.parametrize(
    ("excel_filepath", "excel_df"),
    (
        (excel_filepath, excel_filepath_to_dataframe(excel_filepath))
        for excel_filepath in EXCEL_DATASETS
    ),
)
def test_excel_dataset_from_fileobj_matches_expected(
    excel_filepath: pathlib.Path, excel_df: pd.DataFrame
):
    """Tests that excelDataset, loaded from file object, matches excel loaded by Pandas.

    Checks:
    - __len__
    - get_columns
    - read_streaming_records
    - read_streaming_values
    - read_file_as_dataframe
    """
    # load excelDataset from object
    with open(excel_filepath, **ExcelDataset.READ_ARGS) as excel_file:
        excel_dataset = ExcelDataset(fileobj=excel_file, file_type="xlsx")

        # check that methods match values read by pandas
        assert_dataset_matches_dataframe(excel_dataset, excel_df)
