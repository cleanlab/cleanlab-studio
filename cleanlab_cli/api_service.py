"""
Methods for interacting with the command line server API
1:1 mapping with command_line_api.py
"""
import asyncio
import gzip
import json
import os
from typing import Dict, List, Any

import aiohttp
import requests

from cleanlab_cli.click_helpers import abort, warn
from cleanlab_cli.dataset.schema_types import Schema
from cleanlab_cli import __version__
from cleanlab_cli.types import JSONDict, IDType

base_url = os.environ.get("CLEANLAB_API_BASE_URL", "https://api.cleanlab.ai/api/cli/v0")


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


def initialize_dataset(api_key: str, schema: Schema) -> str:
    request_json = dict(api_key=api_key, schema=schema.to_dict())
    res = requests.post(base_url + "/datasets", json=request_json)
    handle_api_error(res)
    dataset_id: str = res.json()["dataset_id"]
    return dataset_id


def get_existing_ids(api_key: str, dataset_id: str) -> List[IDType]:
    res = requests.get(base_url + f"/datasets/{dataset_id}/ids", json=dict(api_key=api_key))
    handle_api_error(res)
    existing_ids: List[IDType] = res.json()["existing_ids"]
    return existing_ids


def get_dataset_schema(api_key: str, dataset_id: str) -> Schema:
    res = requests.get(base_url + f"/datasets/{dataset_id}/schema", json=dict(api_key=api_key))
    handle_api_error(res)
    schema: Schema = res.json()["schema"]
    return schema


async def upload_rows_async(
    session: aiohttp.ClientSession,
    api_key: str,
    dataset_id: str,
    rows: List[Any],
    columns_json: str,
) -> None:
    url = base_url + f"/datasets/{dataset_id}"
    data = gzip.compress(
        json.dumps(dict(api_key=api_key, rows=json.dumps(rows), columns=columns_json)).encode(
            "utf-8"
        )
    )
    headers = {
        "Content-Type": "application/json",
        "Content-Encoding": "gzip",
    }

    async with session.post(url=url, data=data, headers=headers) as res:
        res_text = await res.read()
        handle_api_error_from_json(json.loads(res_text))


def download_cleanlab_columns(api_key: str, cleanset_id: str, all: bool = False) -> List[List[Any]]:
    """
    Download all rows from specified Cleanlab columns

    :param api_key:
    :param cleanset_id:
    :param all: whether to download all Cleanlab columns or just the clean_label column
    :return: return (rows, id_column)
    """
    res = requests.get(
        base_url + f"/cleansets/{cleanset_id}/columns?all={all}", json=dict(api_key=api_key)
    )
    handle_api_error(res)
    rows: List[List[Any]] = res.json()["rows"]
    return rows


def get_completion_status(api_key: str, dataset_id: str) -> bool:
    res = requests.get(base_url + f"/datasets/{dataset_id}/complete", json=dict(api_key=api_key))
    handle_api_error(res)
    completed: bool = res.json()["complete"]
    return completed


def complete_upload(api_key: str, dataset_id: str) -> None:
    res = requests.patch(base_url + f"/datasets/{dataset_id}/complete", json=dict(api_key=api_key))
    handle_api_error(res)


def get_id_column(api_key: str, cleanset_id: str) -> str:
    res = requests.get(base_url + f"/cleansets/{cleanset_id}/id_column", json=dict(api_key=api_key))
    handle_api_error(res)
    id_column: str = res.json()["id_column"]
    return id_column


def validate_api_key(api_key: str) -> bool:
    res = requests.get(base_url + "/validate", json=dict(api_key=api_key))
    handle_api_error(res)
    valid: bool = res.json()["valid"]
    return valid


def check_client_version() -> bool:
    res = requests.post(base_url + "/check_client_version", json=dict(version=__version__))
    handle_api_error(res)
    valid: bool = res.json()["valid"]
    return valid


def check_dataset_limit(file_size: int, api_key: str, show_warning: bool = False) -> JSONDict:
    res = requests.post(
        base_url + "/check_dataset_limit", json=dict(api_key=api_key, file_size=file_size)
    )
    handle_api_error(res, show_warning=show_warning)
    res_json: JSONDict = res.json()
    return res_json
