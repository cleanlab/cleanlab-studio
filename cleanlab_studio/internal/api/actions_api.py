"""Module for interfacing with the cleanset actions API."""
from typing import Any, List, Optional

import requests

from cleanlab_studio.internal.types import JSONDict
from .api import base_url, cleanset_base_url, construct_headers, handle_api_error_from_json


datasheets_base_url = f"{base_url}/datasheets/v1"


def handle_actions_api_error(res_json: JSONDict) -> None:
    """Handles an error response from the actions API.

    Args:
        res (JSONDict): response from the actions API

    Raises:
        RowIdTypeError: if row ID type does not match type in cleanset
        RowNotFoundError: if row ID is not found in cleanset
        LabelTypeError: if label type does not match type in cleanset
        LabelValueError: if label value is not in the valid set of labels for the cleanset
        ColumnNotFoundError: if column (provided in dataset or cleanlab columns) is not found in cleanset
        TaskTypeError: if this operation is not supported for the task type of
    """
    error_type = res_json.get("error_type")
    error_message = res_json.get("error_message")

    if error_type == "RowIdTypeError":
        raise RowIdTypeError(error_message)
    elif error_type == "RowNotFoundError":
        raise RowNotFoundError(error_message)
    elif error_type == "LabelTypeError":
        raise LabelTypeError(error_message)
    elif error_type == "LabelValueError":
        raise LabelValueError(error_message)
    elif error_type == "ColumnNotFoundError":
        raise ColumnNotFoundError(error_message)
    elif error_type == "TaskTypeError":
        raise TaskTypeError(error_message)
    else:
        # fallback to generic error handler
        handle_api_error_from_json(res_json)

    return


def read_row(
    api_key: str,
    cleanset_id: str,
    row_id: Any,
    dataset_columns: Optional[List[str]],
    cleanlab_columns: Optional[List[str]],
) -> JSONDict:
    """Reads a row from the cleanset.

    Args:
        api_key (str): API key for Cleanlab Studio
        cleanset_id (str): ID of cleanset to read from
        row_id (Any): row ID to get data for
        dataset_columns (Optional[List[str]]): list of dataset columns to include in returned row, defaults to None.
        cleanlab_columns (Optional[List[str]]): list of cleanlab columns to include in returned row, defaults to None.

    Returns:
        JSONDict: dictionary containing key-value pairs for row in cleanset
                  keys of dictionary will be set of dataset and cleanlab columns requested or all columns in dataset and cleanset

    Raises:
        RowIdTypeError: if row ID type does not match type in cleanset
        RowNotFoundError: if row ID is not found in cleanset
        ColumnNotFoundError: if column (provided in dataset or cleanlab columns) is not found in cleanset
        TaskTypeError: if this operation is not supported for the task type of the cleanset
    """
    res = requests.get(
        f"{datasheets_base_url}/{cleanset_id}/row/{row_id}",
        params={
            "dataset_columns": dataset_columns,
            "cleanlab_columns": cleanlab_columns,
        },
        headers=construct_headers(api_key),
    )
    res_json = res.json()
    handle_actions_api_error(res_json)

    return res_json
