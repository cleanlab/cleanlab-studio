import gzip
import json
import os
import requests
from typing import Optional, List, Any

from cleanlab_studio.version import __version__
from cleanlab_studio.internal.types import JSONDict, IDType
from cleanlab_studio.internal.schema import Schema
from cleanlab_studio.errors import APIError, UnsupportedVersionError, AuthError


base_url = os.environ.get("CLEANLAB_API_BASE_URL", "https://api.cleanlab.ai/api/cli/v0")


def _construct_headers(
    api_key: Optional[str], content_type: Optional[str] = "application/json"
) -> JSONDict:
    retval = dict()
    if api_key:
        retval["Authorization"] = f"bearer {api_key}"
    if content_type:
        retval["Content-Type"] = content_type
    return retval


def check_client_version() -> None:
    res = requests.post(base_url + "/check_client_version", json=dict(version=__version__))
    handle_api_error(res)
    valid: bool = res.json()["valid"]
    if not valid:
        raise UnsupportedVersionError


def handle_api_error(res: requests.Response) -> None:
    handle_api_error_from_json(res.json())


def handle_api_error_from_json(res_json: JSONDict) -> None:
    if "code" in res_json and "description" in res_json:  # AuthError or UserQuotaError format
        if res_json["code"] == "user_soft_quota_exceeded":
            pass  # soft quota limit is going away soon, so ignore it
        else:
            raise APIError(res_json["description"])
    if res_json.get("error", None) is not None:
        raise APIError(res_json["error"])


def validate_api_key(api_key: str) -> None:
    res = requests.get(
        base_url + "/validate", json=dict(api_key=api_key), headers=_construct_headers(api_key)
    )
    handle_api_error(res)
    valid: bool = res.json()["valid"]
    if not valid:
        raise AuthError


def initialize_dataset(api_key: str, schema: Schema) -> str:
    request_json = dict(schema=schema.to_dict())
    res = requests.post(
        base_url + "/datasets", json=request_json, headers=_construct_headers(api_key)
    )
    handle_api_error(res)
    dataset_id: str = res.json()["dataset_id"]
    return dataset_id


def get_existing_ids(api_key: str, dataset_id: str) -> List[IDType]:
    res = requests.get(
        base_url + f"/datasets/{dataset_id}/ids",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    existing_ids: List[IDType] = res.json()["existing_ids"]
    return existing_ids


def upload_rows(api_key: str, dataset_id: str, rows: List[Any], columns: List[str]) -> None:
    url = base_url + f"/datasets/{dataset_id}"
    data = gzip.compress(
        json.dumps(dict(rows=json.dumps(rows), columns=json.dumps(columns))).encode("utf-8")
    )
    headers = _construct_headers(api_key)
    headers["Content-Encoding"] = "gzip"
    res = requests.post(url, data=data, headers=headers)
    handle_api_error(res)


def complete_upload(api_key: str, dataset_id: str) -> None:
    res = requests.patch(
        base_url + f"/datasets/{dataset_id}/complete",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)


def download_cleanlab_columns(api_key: str, cleanset_id: str, all: bool = False) -> List[List[Any]]:
    """
    Download all rows from specified Cleanlab columns

    :param api_key:
    :param cleanset_id:
    :param all: whether to download all Cleanlab columns or just the clean_label column
    :return: return (rows, id_column)
    """
    res = requests.get(
        base_url + f"/cleansets/{cleanset_id}/columns?all={all}",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    rows: List[List[Any]] = res.json()["rows"]
    return rows


def get_id_column(api_key: str, cleanset_id: str) -> str:
    res = requests.get(
        base_url + f"/cleansets/{cleanset_id}/id_column",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    id_column: str = res.json()["id_column"]
    return id_column


def get_project_of_cleanset(api_key: str, cleanset_id: str) -> str:
    res = requests.get(
        base_url + f"/cleansets/{cleanset_id}/project",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    project_id: str = res.json()["project_id"]
    return project_id


def get_dataset_of_project(api_key: str, project_id: str) -> str:
    res = requests.get(
        base_url + f"/projects/{project_id}/dataset",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    dataset_id: str = res.json()["dataset_id"]
    return dataset_id


def get_label_column_of_project(api_key: str, project_id: str) -> str:
    res = requests.get(
        base_url + f"/projects/{project_id}/label_column",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    label_column: str = res.json()["label_column"]
    return label_column


def get_dataset_schema(api_key: str, dataset_id: str) -> Schema:
    res = requests.get(
        base_url + f"/datasets/{dataset_id}/schema",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    schema: Schema = Schema.from_dict(res.json()["schema"])
    return schema


def get_presigned_posts(
    api_key: str, dataset_id: str, filepaths: List[str], row_ids: List[str], media_type: str
) -> JSONDict:
    res = requests.get(
        base_url + "/media_upload/presigned_posts",
        json=dict(dataset_id=dataset_id, filepaths=filepaths, row_ids=row_ids, type=media_type),
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    res_json: JSONDict = res.json()
    return res_json
