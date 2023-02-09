import pathlib
from typing import List

import pandas as pd
import pytest

from cleanlab_studio.cli.classes.json_dataset import JsonDataset
from .constants import DATASETS_DIR
from .utils import assert_dataset_matches_dataframe, json_filepath_to_dataframe


json_DATASETS: List[pathlib.Path] = [*DATASETS_DIR.glob(r"*.json")]


@pytest.mark.parametrize(
    ("json_filepath", "json_df"),
    ((json_filepath, json_filepath_to_dataframe(json_filepath)) for json_filepath in json_DATASETS),
)
def test_json_dataset_from_filepath_matches_expected(
    json_filepath: pathlib.Path, json_df: pd.DataFrame
):
    """Tests that JsonDataset, loaded from filepath, matches json loaded by Pandas.

    Checks:
    - __len__
    - get_columns
    - read_streaming_records
    - read_streaming_values
    - read_file_as_dataframe
    """
    # load jsonDataset from filepath
    json_dataset = JsonDataset(filepath=str(json_filepath))

    # check that methods match values read by pandas
    assert_dataset_matches_dataframe(json_dataset, json_df, ignore_id=True)


@pytest.mark.parametrize(
    ("json_filepath", "json_df"),
    ((json_filepath, json_filepath_to_dataframe(json_filepath)) for json_filepath in json_DATASETS),
)
def test_json_dataset_from_fileobj_matches_expected(
    json_filepath: pathlib.Path, json_df: pd.DataFrame
):
    """Tests that JsonDataset, loaded from file object, matches json loaded by Pandas.

    Checks:
    - __len__
    - get_columns
    - read_streaming_records
    - read_streaming_values
    - read_file_as_dataframe
    """
    # load jsonDataset from object
    with open(json_filepath, **JsonDataset.READ_ARGS) as json_file:
        json_dataset = JsonDataset(fileobj=json_file)

        # check that methods match values read by pandas
        assert_dataset_matches_dataframe(json_dataset, json_df, ignore_id=True)
