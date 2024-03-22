"""
Methods for interacting with the command line server API
1:1 mapping with command_line_api.py
"""

import os
from typing import List, Optional

import requests
import pandas as pd

from cleanlab_studio.version import __version__
from cleanlab_studio.cli.click_helpers import abort, warn
from cleanlab_studio.cli.types import JSONDict

base_url = os.environ.get("CLEANLAB_API_BASE_URL", "https://api.cleanlab.ai/api")
cli_base_url = f"{base_url}/cli/v0"


def _construct_headers(
    api_key: Optional[str],
    content_type: Optional[str] = "application/json",
) -> JSONDict:
    retval = dict()
    if api_key:
        retval["Authorization"] = f"bearer {api_key}"
    if content_type:
        retval["Content-Type"] = content_type
    retval["Client-Type"] = "cli"
    retval["User-Agent"] = f"cleanlab-studio/v{__version__}"
    return retval
    return retval


def handle_api_error(res: requests.Response, show_warning: bool = False) -> None:
    handle_api_error_from_json(res.json(), show_warning)


def handle_api_error_from_json(res_json: JSONDict, show_warning: bool = False) -> None:
    if "code" in res_json and "description" in res_json:  # AuthError or UserQuotaError format
        if res_json["code"] == "user_soft_quota_exceeded":
            if show_warning:
                warn(res_json["description"])
        else:
            abort(res_json["description"])
    if res_json.get("error", None) is not None:
        abort(res_json["error"])


def download_cleanlab_columns(
    api_key: str,
    cleanset_id: str,
    include_cleanlab_columns: bool = False,
    include_project_details: bool = False,
) -> pd.DataFrame:
    """
    Download all rows from specified Cleanlab columns

    :param api_key:
    :param cleanset_id:
    :param all: whether to download all Cleanlab columns or just the clean_label column
    :return: return (rows, id_column)
    """
    res = requests.get(
        cli_base_url
        + f"/cleansets/{cleanset_id}/columns?"
        + f"include_cleanlab_columns={include_cleanlab_columns}"
        + f"&include_project_details={include_project_details}",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    cleanset_json: str = res.json()["cleanset_json"]
    cleanset_df: pd.DataFrame = pd.read_json(cleanset_json, orient="table")
    id_col = get_id_column(api_key, cleanset_id)
    cleanset_df.rename(columns={"id": id_col}, inplace=True)
    return cleanset_df


def get_completion_status(api_key: str, dataset_id: str) -> bool:
    res = requests.get(
        cli_base_url + f"/datasets/{dataset_id}/complete",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    completed: bool = res.json()["complete"]
    return completed


def complete_upload(api_key: str, dataset_id: str) -> None:
    res = requests.patch(
        cli_base_url + f"/datasets/{dataset_id}/complete",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)


def get_id_column(api_key: str, cleanset_id: str) -> str:
    res = requests.get(
        cli_base_url + f"/cleansets/{cleanset_id}/id_column",
        headers={"Authorization": f"bearer {api_key}"},
    )
    handle_api_error(res)
    id_column: str = res.json()["id_column"]
    return id_column


def validate_api_key(api_key: str) -> bool:
    res = requests.get(
        cli_base_url + "/validate", json=dict(api_key=api_key), headers=_construct_headers(api_key)
    )
    handle_api_error(res)
    valid: bool = res.json()["valid"]
    return valid


def check_client_version() -> bool:
    res = requests.post(cli_base_url + "/check_client_version", json=dict(version=__version__))
    handle_api_error(res)
    valid: bool = res.json()["valid"]
    return valid


def check_dataset_limit(
    api_key: str, file_size: int, modality: str, show_warning: bool = False
) -> JSONDict:
    res = requests.post(
        cli_base_url + "/check_dataset_limit",
        json=dict(file_size=file_size, modality=modality),
        headers=_construct_headers(api_key),
    )
    handle_api_error(res, show_warning=show_warning)
    res_json: JSONDict = res.json()
    return res_json


def get_presigned_posts(
    api_key: str, dataset_id: str, filepaths: List[str], row_ids: List[str], media_type: str
) -> JSONDict:
    res = requests.get(
        cli_base_url + "/media_upload/presigned_posts",
        json=dict(dataset_id=dataset_id, filepaths=filepaths, row_ids=row_ids, type=media_type),
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    res_json: JSONDict = res.json()
    return res_json
