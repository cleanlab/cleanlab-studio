import os
import sys
import subprocess
import platform
import time
from typing import Any, Callable, List, Optional, Tuple, Dict
import functools
import sys
import traceback
import json
import contextlib

from cleanlab_studio.errors import APIError

import requests
from tqdm import tqdm
import pandas as pd
import numpy as np
import numpy.typing as npt

from cleanlab_studio.internal.types import JSONDict
from cleanlab_studio.version import __version__

base_url = os.environ.get("CLEANLAB_API_BASE_URL", "https://api.cleanlab.ai/api")
cli_base_url = f"{base_url}/cli/v0"
upload_base_url = f"{base_url}/upload/v0"
dataset_base_url = f"{base_url}/datasets"
project_base_url = f"{base_url}/projects"
cleanset_base_url = f"{base_url}/cleansets"


def telemetry(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to send stack trace to backend if an exception is raised

    Can only use on functions whose first arg is api_key
    """

    @functools.wraps(func)
    def tracked_func(api_key: str, *args: Any, **kwargs: Any) -> Any:
        user_info = get_basic_info()
        user_info["func_name"] = func.__name__
        try:
            result = func(api_key, *args, **kwargs)
            _ = requests.post(
                f"{cli_base_url}/telemetry",
                json=user_info,
                headers=_construct_headers(api_key),
            )
            return result
        except Exception as err:
            with contextlib.suppress(Exception):
                stacks = traceback.format_stack()
                # only send stack trace for cleanlab functions
                cleanlab_traceback = [line for line in stacks if "cleanlab" in line.lower()]
                err_info = {
                    "stack_trace": "\n".join(cleanlab_traceback),
                }
                user_info.update(err_info)
                _ = requests.post(
                    f"{cli_base_url}/telemetry",
                    json=user_info,
                    headers=_construct_headers(api_key),
                )
            raise err

    return tracked_func


def get_basic_info() -> Dict[str, Any]:
    user_info: Dict[str, Any] = {}
    # get OS
    user_info["os"] = platform.system()
    user_info["os_release"] = platform.release()
    # get python version
    user_info["python_version"] = sys.version
    # get CLI version and dependencies
    studio_info = (
        subprocess.check_output(
            [
                sys.executable,
                "-m",
                "pip",
                "show",
                "cleanlab-studio",
            ]
        )
        .decode("utf-8")
        .split("\n")
    )
    dependencies = []
    for line in studio_info:
        if line.startswith("Version:"):
            user_info["cli_version"] = line.split(" ")[1]
        elif line.startswith("Requires:"):
            dependencies = line.split(": ")[1].split(", ")
    all_packages = (
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode("utf-8").split("\n")
    )
    package_versions = dict([package.split("==") for package in all_packages if "==" in package])
    user_info["dependencies"] = {
        dependency: package_versions.get(dependency) for dependency in dependencies
    }
    return user_info


def _construct_headers(
    api_key: Optional[str], content_type: Optional[str] = "application/json"
) -> JSONDict:
    retval = dict()
    if api_key:
        retval["Authorization"] = f"bearer {api_key}"
    if content_type:
        retval["Content-Type"] = content_type
    return retval


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


def validate_api_key(api_key: str) -> bool:
    res = requests.get(
        cli_base_url + "/validate",
        json=dict(api_key=api_key),
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    valid: bool = res.json()["valid"]
    return valid


def is_valid_client_version() -> bool:
    res = requests.post(cli_base_url + "/check_client_version", json=dict(version=__version__))
    handle_api_error(res)
    valid: bool = res.json()["valid"]
    return valid


def initialize_upload(
    api_key: str, filename: str, file_type: str, file_size: int
) -> Tuple[str, List[int], List[str]]:
    res = requests.get(
        f"{upload_base_url}/initialize",
        params=dict(size_in_bytes=str(file_size), filename=filename, file_type=file_type),
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    upload_id: str = res.json()["upload_id"]
    part_sizes: List[int] = res.json()["part_sizes"]
    presigned_posts: List[str] = res.json()["presigned_posts"]
    return upload_id, part_sizes, presigned_posts


def complete_file_upload(api_key: str, upload_id: str, upload_parts: List[JSONDict]) -> None:
    request_json = dict(upload_id=upload_id, upload_parts=upload_parts)
    res = requests.post(
        f"{upload_base_url}/complete",
        json=request_json,
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)


def get_proposed_schema(api_key: str, upload_id: str) -> JSONDict:
    res = requests.get(
        f"{upload_base_url}/proposed_schema",
        params=dict(upload_id=upload_id),
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    res_json: JSONDict = res.json()
    return res_json


def confirm_schema(
    api_key: str,
    schema: Optional[JSONDict],
    upload_id: str,
) -> None:
    request_json = dict(schema=schema, upload_id=upload_id)
    res = requests.post(
        f"{upload_base_url}/confirm_schema",
        json=request_json,
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)


def get_ingestion_status(api_key: str, upload_id: str) -> JSONDict:
    res = requests.get(
        f"{upload_base_url}/ingestion_status",
        params=dict(upload_id=upload_id),
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    res_json: JSONDict = res.json()
    return res_json


def get_dataset_id(api_key: str, upload_id: str) -> JSONDict:
    res = requests.get(
        f"{upload_base_url}/dataset_id",
        params=dict(upload_id=upload_id),
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    res_json: JSONDict = res.json()
    return res_json


def get_project_of_cleanset(api_key: str, cleanset_id: str) -> str:
    res = requests.get(
        cli_base_url + f"/cleansets/{cleanset_id}/project",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    project_id: str = res.json()["project_id"]
    return project_id


def get_label_column_of_project(api_key: str, project_id: str) -> str:
    res = requests.get(
        cli_base_url + f"/projects/{project_id}/label_column",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    label_column: str = res.json()["label_column"]
    return label_column


@telemetry
def download_cleanlab_columns(api_key: str, cleanset_id: str, all: bool = False) -> pd.DataFrame:
    """
    Download all rows from specified Cleanlab columns

    :param api_key:
    :param cleanset_id:
    :param all: whether to download all Cleanlab columns or just the clean_label column
    :return: return (rows, id_column)
    """
    res = requests.get(
        cli_base_url + f"/cleansets/{cleanset_id}/columns?all={all}",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    cleanset_json: str = res.json()["cleanset_json"]
    cleanset_df: pd.DataFrame = pd.read_json(cleanset_json, orient="table")
    id_col = get_id_column(api_key, cleanset_id)
    cleanset_df.rename(columns={"id": id_col}, inplace=True)
    return cleanset_df


def download_numpy(api_key: str, cleanset_id: str, name: str) -> npt.NDArray[np.float_]:
    res = requests.get(
        cli_base_url + f"/cleansets/{cleanset_id}/{name}",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    res_json: JSONDict = res.json()
    if res_json["success"]:
        np_data: npt.NDArray[np.float_] = np.array(res_json[name])
        return np_data
    raise APIError(f"{name} for cleanset {cleanset_id} not found")


def get_id_column(api_key: str, cleanset_id: str) -> str:
    res = requests.get(
        cli_base_url + f"/cleansets/{cleanset_id}/id_column",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    id_column: str = res.json()["id_column"]
    return id_column


def get_dataset_of_project(api_key: str, project_id: str) -> str:
    res = requests.get(
        cli_base_url + f"/projects/{project_id}/dataset",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    dataset_id: str = res.json()["dataset_id"]
    return dataset_id


def get_dataset_schema(api_key: str, dataset_id: str) -> JSONDict:
    res = requests.get(
        cli_base_url + f"/datasets/{dataset_id}/schema",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    schema: JSONDict = res.json()["schema"]
    return schema


def get_dataset_details(api_key: str, dataset_id: str) -> JSONDict:
    res = requests.get(
        dataset_base_url + f"/details/{dataset_id}",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    dataset_details: JSONDict = res.json()
    return dataset_details


def clean_dataset(
    api_key: str,
    dataset_id: str,
    project_name: str,
    task_type: str,
    modality: str,
    model_type: str,
    label_column: str,
    feature_columns: List[str],
    text_column: Optional[str],
) -> str:
    request_json = dict(
        name=project_name,
        dataset_id=dataset_id,
        tasktype=task_type,
        modality=modality,
        model_type=model_type,
        label_column=label_column,
        feature_columns=feature_columns,
        text_column=text_column,
    )
    res = requests.post(
        project_base_url + f"/clean",
        headers=_construct_headers(api_key),
        json=request_json,
    )
    handle_api_error(res)
    project_id = res.json()["project_id"]
    return str(project_id)


def get_latest_cleanset_id(api_key: str, project_id: str) -> str:
    res = requests.get(
        cleanset_base_url + f"/project/{project_id}/latest_cleanset_id",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    cleanset_id = res.json()["cleanset_id"]
    return str(cleanset_id)


def get_cleanset_status(api_key: str, cleanset_id: str) -> JSONDict:
    res = requests.get(
        cleanset_base_url + f"/{cleanset_id}/status",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    status: JSONDict = res.json()
    return status


def delete_project(api_key: str, project_id: str) -> None:
    res = requests.delete(project_base_url + f"/{project_id}", headers=_construct_headers(api_key))
    handle_api_error(res)


def poll_progress(
    progress_id: str, request_function: Callable[[str], JSONDict], description: str
) -> JSONDict:
    with tqdm(total=1, desc=description, bar_format="{desc}: {percentage:3.0f}%|{bar}|") as pbar:
        res = request_function(progress_id)
        while res["status"] != "complete":
            if res["status"] == "error":
                raise APIError(res["error_message"])
            pbar.update(float(res["progress"]) - pbar.n)
            time.sleep(0.5)
            res = request_function(progress_id)
        pbar.update(float(1) - pbar.n)
    return res
