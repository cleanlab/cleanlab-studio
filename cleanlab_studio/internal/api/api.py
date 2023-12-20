import io
import os
import time
from typing import Callable, cast, List, Optional, Tuple, Dict, Union, Any
from cleanlab_studio.errors import APIError

import requests
from tqdm import tqdm
import pandas as pd
import numpy as np
import numpy.typing as npt

try:
    import snowflake

    snowflake_exists = True
except ImportError:
    snowflake_exists = False

try:
    import pyspark.sql

    pyspark_exists = True
except ImportError:
    pyspark_exists = False

from cleanlab_studio.internal.types import JSONDict
from cleanlab_studio.version import __version__


base_url = os.environ.get("CLEANLAB_API_BASE_URL", "https://api.cleanlab.ai/api")
cli_base_url = f"{base_url}/cli/v0"
upload_base_url = f"{base_url}/upload/v0"
dataset_base_url = f"{base_url}/datasets"
project_base_url = f"{base_url}/projects"
cleanset_base_url = f"{base_url}/cleansets"
model_base_url = f"{base_url}/v1/deployment"
tlm_base_url = f"{base_url}/v0/trustworthy_llm"


def _construct_headers(
    api_key: Optional[str], content_type: Optional[str] = "application/json"
) -> JSONDict:
    retval = dict()
    if api_key:
        retval["Authorization"] = f"bearer {api_key}"
    if content_type:
        retval["Content-Type"] = content_type
    retval["Client-Type"] = "python-api"
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


def download_cleanlab_columns(
    api_key: str,
    cleanset_id: str,
    include_cleanlab_columns: bool = True,
    include_project_details: bool = False,
    to_spark: bool = False,
) -> Any:
    """
    Download all rows from specified Cleanlab columns

    :param api_key:
    :param cleanset_id:
    :param include_cleanlab_columns: whether to download all Cleanlab columns or just the clean_label column
    :param include_project_details: whether to download columns related to project status such as resolved rows, actions taken, etc.
    :return: return a dataframe, either pandas or spark. Type is Any because don't want to require spark installed
    """
    res = requests.get(
        cli_base_url + f"/cleansets/{cleanset_id}/columns",
        params=dict(
            to_spark=to_spark,
            include_cleanlab_columns=include_cleanlab_columns,
            include_project_details=include_project_details,
        ),
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    id_col = get_id_column(api_key, cleanset_id)
    cleanset_json: str = res.json()["cleanset_json"]
    if to_spark:
        if not pyspark_exists:
            raise ImportError(
                "pyspark is not installed. Please install pyspark to download cleanlab columns as a pyspark DataFrame."
            )
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        rdd = spark.sparkContext.parallelize([cleanset_json])
        cleanset_pyspark: pyspark.sql.DataFrame = spark.read.json(rdd)
        cleanset_pyspark = cleanset_pyspark.withColumnRenamed("id", id_col)
        cleanset_pyspark = cleanset_pyspark.sort(id_col)
        return cleanset_pyspark

    cleanset_pd: pd.DataFrame = pd.read_json(cleanset_json, orient="table")
    cleanset_pd.rename(columns={"id": id_col}, inplace=True)
    cleanset_pd.sort_values(by=id_col, inplace=True)
    return cleanset_pd


def download_array(
    api_key: str, cleanset_id: str, name: str
) -> Union[npt.NDArray[np.float_], pd.DataFrame]:
    res = requests.get(
        cli_base_url + f"/cleansets/{cleanset_id}/{name}",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    res_json: JSONDict = res.json()
    if res_json["success"]:
        if res_json["array_type"] == "numpy":
            np_data: npt.NDArray[np.float_] = np.array(res_json[name])
            return np_data
        pd_data: pd.DataFrame = pd.read_json(res_json[name], orient="records")
        return pd_data
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


def get_dataset_details(api_key: str, dataset_id: str, task_type: str) -> JSONDict:
    res = requests.get(
        project_base_url + f"/dataset_details/{dataset_id}",
        params=dict(tasktype=task_type),
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


def upload_predict_batch(api_key: str, model_id: str, batch: io.StringIO) -> str:
    """Uploads prediction batch and returns query ID."""
    url = f"{model_base_url}/{model_id}/upload"
    res = requests.post(
        url,
        headers=_construct_headers(api_key),
    )

    handle_api_error(res)
    presigned_url = res.json()["upload_url"]
    query_id: str = res.json()["query_id"]

    requests.post(presigned_url["url"], data=presigned_url["fields"], files={"file": batch})

    return query_id


def start_prediction(api_key: str, model_id: str, query_id: str) -> None:
    """Starts prediction for query."""
    res = requests.post(
        f"{model_base_url}/{model_id}/predict/{query_id}",
        headers=_construct_headers(api_key),
    )

    handle_api_error(res)


def get_prediction_status(api_key: str, query_id: str) -> Dict[str, str]:
    """Gets status of model prediction query. Returns status, and optionally the result url or error message."""
    res = requests.get(
        f"{model_base_url}/predict/{query_id}",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)

    return cast(Dict[str, str], res.json())


def get_deployed_model_info(api_key: str, model_id: str) -> Dict[str, str]:
    """Get info about deployed model, including model id, name, cleanset id, dataset id, projectid, updated_at, status, and tasktype"""
    res = requests.get(
        f"{model_base_url}/{model_id}",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)

    return cast(Dict[str, str], res.json())


def tlm_prompt(
    api_key: str,
    prompt: str,
    quality_preset: str,
    options: Optional[JSONDict],
) -> JSONDict:
    """
    Prompt Trustworthy Language Model with a question, and get back its answer along with a confidence score

    Args:
        api_key (str): studio API key for auth
        prompt (str): prompt for TLM to respond to
        quality_preset (str): quality preset to use to generate response

    Returns:
        JSONDict: dictionary with TLM response and confidence score
    """
    res = requests.post(
        f"{tlm_base_url}/prompt",
        json=dict(prompt=prompt, quality=quality_preset, options=options or {}),
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    return cast(JSONDict, res.json())


def tlm_get_confidence_score(
    api_key: str,
    prompt: str,
    response: str,
    quality_preset: str,
    options: Optional[JSONDict],
) -> JSONDict:
    """
    Query Trustworthy Language Model for a confidence score for the prompt-response pair.

    Args:
        api_key (str): studio API key for auth
        prompt (str): prompt for TLM to get confidence score for
        response (str): response for TLM to get confidence score for
        quality_preset (str): quality preset to use to generate confidence score

    Returns:
        JSONDict: dictionary with TLM confidence score
    """
    res = requests.post(
        f"{tlm_base_url}/get_confidence_score",
        json=dict(prompt=prompt, response=response, quality=quality_preset, options=options or {}),
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    return cast(JSONDict, res.json())
