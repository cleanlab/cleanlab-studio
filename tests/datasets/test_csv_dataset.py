import pathlib
from typing import List

import pandas as pd
import pytest

from cleanlab_studio.cli.classes.csv_dataset import CsvDataset
from .constants import DATASETS_DIR
from .utils import assert_dataset_matches_dataframe, csv_filepath_to_dataframe


CSV_DATASETS: List[pathlib.Path] = [*DATASETS_DIR.glob(r"*.csv")]


@pytest.mark.parametrize(
    ("csv_filepath", "csv_df"),
    ((csv_filepath, csv_filepath_to_dataframe(csv_filepath)) for csv_filepath in CSV_DATASETS),
)
def test_csv_dataset_from_filepath_matches_expected(
    csv_filepath: pathlib.Path, csv_df: pd.DataFrame
):
    """Tests that CsvDataset, loaded from filepath, matches CSV loaded by Pandas.

    Checks:
    - __len__
    - get_columns
    - read_streaming_records
    - read_streaming_values
    - read_file_as_dataframe
    """
    # load CsvDataset from filepath
    csv_dataset = CsvDataset(filepath=str(csv_filepath))

    # check that methods match values read by pandas
    assert_dataset_matches_dataframe(csv_dataset, csv_df)


@pytest.mark.parametrize(
    ("csv_filepath", "csv_df"),
    ((csv_filepath, csv_filepath_to_dataframe(csv_filepath)) for csv_filepath in CSV_DATASETS),
)
def test_csv_dataset_from_fileobj_matches_expected(
    csv_filepath: pathlib.Path, csv_df: pd.DataFrame
):
    """Tests that CsvDataset, loaded from file object, matches CSV loaded by Pandas.

    Checks:
    - __len__
    - get_columns
    - read_streaming_records
    - read_streaming_values
    - read_file_as_dataframe
    """
    # load CsvDataset from object
    with open(csv_filepath, **CsvDataset.READ_ARGS) as csv_file:
        csv_dataset = CsvDataset(fileobj=csv_file)

        # check that methods match values read by pandas
        assert_dataset_matches_dataframe(csv_dataset, csv_df)
