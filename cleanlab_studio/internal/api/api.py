from __future__ import annotations

import asyncio
import io
import os
import ssl
import time
import warnings
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import aiohttp
import aiohttp.client_exceptions
import numpy as np
import numpy.typing as npt
import pandas as pd
import requests
from tqdm import tqdm

from cleanlab_studio.errors import (
    APIError,
    IngestionError,
    InvalidProjectConfiguration,
    RateLimitError,
    TlmBadRequest,
    TlmPartialSuccess,
    TlmServerError,
)
from cleanlab_studio.internal.tlm.concurrency import TlmRateHandler

try:
    import snowflake
    import snowflake.snowpark as snowpark

    snowflake_exists = True
except ImportError:
    snowflake_exists = False

try:
    import pyspark.sql

    pyspark_exists = True
except ImportError:
    pyspark_exists = False

from cleanlab_studio.errors import NotInstalledError
from cleanlab_studio.internal.api.api_helper import (
    check_uuid_well_formed,
    check_valid_csv_filename,
)
from cleanlab_studio.internal.types import JSONDict, SchemaOverride, TLMQualityPreset
from cleanlab_studio.version import __version__

base_url = os.environ.get("CLEANLAB_API_BASE_URL", "https://api.cleanlab.ai/api")
cli_base_url = f"{base_url}/cli/v0"
upload_base_url = f"{base_url}/upload/v1"
dataset_base_url = f"{base_url}/datasets"
enrichment_base_url = f"{base_url}/enrichment/v0"
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
    handle_api_error_from_json(res.json(), res.status_code)


def handle_api_error_from_json(res_json: JSONDict, status_code: Optional[int] = None) -> None:
    if "code" in res_json and "description" in res_json:  # AuthError or UserQuotaError format
        if res_json["code"] == "user_soft_quota_exceeded":
            pass  # soft quota limit is going away soon, so ignore it
        else:
            raise APIError(res_json["description"])

    if isinstance(res_json, dict) and res_json.get("error", None) is not None:
        error = res_json["error"]
        if (
            status_code == 422
            and isinstance(error, dict)
            and error.get("code", None) == "UNSUPPORTED_PROJECT_CONFIGURATION"
        ):
            raise InvalidProjectConfiguration(error["description"])
        raise APIError(res_json["error"])

    if status_code != 200:
        raise APIError(f"API call failed with status code {status_code}")


def handle_rate_limit_error_from_resp(resp: aiohttp.ClientResponse) -> None:
    """Catches 429 (rate limit) errors."""
    if resp.status == 429:
        raise RateLimitError(
            f"Rate limit exceeded on {resp.url}", int(resp.headers.get("Retry-After", 0))
        )


async def handle_tlm_client_error_from_resp(
    resp: aiohttp.ClientResponse, batch_index: Optional[int]
) -> None:
    """Catches 4XX (client error) errors."""
    if 400 <= resp.status < 500:
        try:
            res_json = await resp.json()
            error_message = res_json["error"]
            retryable = False
        except Exception:
            error_message = "TLM query failed. Please try again and contact support@cleanlab.ai if the problem persists."
            retryable = True
        if batch_index is not None:
            error_message = f"Error executing query at index {batch_index}:\n{error_message}"

        raise TlmBadRequest(error_message, retryable)


async def handle_tlm_api_error_from_resp(
    resp: aiohttp.ClientResponse, batch_index: Optional[int]
) -> None:
    """Catches 5XX (server error) errors."""
    if 500 <= resp.status < 600:
        try:
            res_json = await resp.json()
            error_message = res_json["error"]
        except Exception:
            error_message = "TLM query failed. Please try again and contact support@cleanlab.ai if the problem persists."

        if batch_index is not None:
            error_message = f"Error executing query at index {batch_index}:\n{error_message}"

        raise TlmServerError(error_message, resp.status)


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
    res = requests.post(
        f"{upload_base_url}/file/initialize",
        json=dict(size_in_bytes=str(file_size), filename=filename, file_type=file_type),
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    upload_id: str = res.json()["upload_id"]
    part_sizes: List[int] = res.json()["part_sizes"]
    presigned_posts: List[str] = res.json()["presigned_posts"]
    return upload_id, part_sizes, presigned_posts


def complete_file_upload(api_key: str, upload_id: str, upload_parts: List[JSONDict]) -> None:
    check_uuid_well_formed(upload_id, "upload ID")
    request_json = dict(upload_id=upload_id, upload_parts=upload_parts)
    res = requests.post(
        f"{upload_base_url}/file/complete",
        json=request_json,
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)


def confirm_upload(
    api_key: str,
    upload_id: str,
    schema_overrides: Optional[List[SchemaOverride]],
) -> None:
    check_uuid_well_formed(upload_id, "upload ID")
    request_json = dict(upload_id=upload_id, schema_overrides=schema_overrides)
    res = requests.post(
        f"{upload_base_url}/confirm",
        json=request_json,
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)


def start_url_upload(
    api_key: str,
    url: str,
    schema_overrides: Optional[List[SchemaOverride]],
) -> str:
    res = requests.post(
        f"{upload_base_url}/url/initialize",
        json=dict(url=url, schema_overrides=schema_overrides),
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)

    upload_id = res.json()["upload_id"]
    return cast(str, upload_id)


def start_bigquery_upload(
    api_key: str,
    bigquery_project: str,
    bigquery_dataset_id: str,
    bigquery_table_id: str,
    schema_overrides: Optional[List[SchemaOverride]],
) -> str:
    res = requests.post(
        f"{upload_base_url}/bigquery/initialize",
        json=dict(
            bigquery_project=bigquery_project,
            bigquery_dataset_id=bigquery_dataset_id,
            bigquery_table_id=bigquery_table_id,
            schema_overrides=schema_overrides,
        ),
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)

    upload_id = res.json()["upload_id"]
    return cast(str, upload_id)


def get_bigquery_principal_account(api_key: str) -> str:
    res = requests.get(
        f"{upload_base_url}/bigquery/principal_account",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    principal_account = res.json()["principal_account"]
    return str(principal_account)


def update_schema(
    api_key: str,
    dataset_id: str,
    schema_overrides: List[SchemaOverride],
) -> None:
    check_uuid_well_formed(dataset_id, "dataset ID")
    request_json = dict(dataset_id=dataset_id, schema_updates=schema_overrides)
    res = requests.patch(
        f"{upload_base_url}/schema",
        json=request_json,
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)


def get_ingestion_status(api_key: str, upload_id: str) -> JSONDict:
    check_uuid_well_formed(upload_id, "upload ID")
    res = requests.get(
        f"{upload_base_url}/total_progress",
        params=dict(upload_id=upload_id),
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    res_json: JSONDict = res.json()
    return res_json


def get_dataset_id(api_key: str, upload_id: str) -> JSONDict:
    check_uuid_well_formed(upload_id, "upload ID")
    res = requests.get(
        f"{upload_base_url}/dataset_id",
        params=dict(upload_id=upload_id),
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    res_json: JSONDict = res.json()
    return res_json


def get_project_of_cleanset(api_key: str, cleanset_id: str) -> str:
    check_uuid_well_formed(cleanset_id, "cleanset ID")
    res = requests.get(
        cli_base_url + f"/cleansets/{cleanset_id}/project",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    project_id: str = res.json()["project_id"]
    return project_id


def get_label_column_of_project(api_key: str, project_id: str) -> str:
    check_uuid_well_formed(project_id, "project ID")
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
    check_uuid_well_formed(cleanset_id, "cleanset ID")
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
            raise NotInstalledError(
                "pyspark is not installed. Please install pyspark to download cleanlab columns as a pyspark DataFrame."
            )
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        rdd = spark.sparkContext.parallelize([cleanset_json])
        cleanset_pyspark: pyspark.sql.DataFrame = spark.read.json(rdd)
        cleanset_pyspark = cleanset_pyspark.withColumnRenamed("id", id_col)
        cleanset_pyspark = cleanset_pyspark.sort(id_col)
        return cleanset_pyspark

    cleanset_json_io = io.StringIO(cleanset_json)
    cleanset_pd: pd.DataFrame = pd.read_json(cleanset_json_io, orient="table")
    cleanset_pd.rename(columns={"id": id_col}, inplace=True)
    cleanset_pd.sort_values(by=id_col, inplace=True, ignore_index=True)
    return cleanset_pd


def download_array(
    api_key: str, cleanset_id: str, name: str
) -> Union[npt.NDArray[np.float64], pd.DataFrame]:
    check_uuid_well_formed(cleanset_id, "cleanset ID")
    res = requests.get(
        cli_base_url + f"/cleansets/{cleanset_id}/{name}",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    res_json: JSONDict = res.json()
    if res_json["success"]:
        if res_json["array_type"] == "numpy":
            np_data: npt.NDArray[np.float64] = np.array(res_json[name])
            return np_data
        pd_data: pd.DataFrame = pd.read_json(StringIO(res_json[name]), orient="records")
        return pd_data
    raise APIError(f"{name} for cleanset {cleanset_id} not found")


def get_id_column(api_key: str, cleanset_id: str) -> str:
    check_uuid_well_formed(cleanset_id, "cleanset ID")
    res = requests.get(
        cli_base_url + f"/cleansets/{cleanset_id}/id_column",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    id_column: str = res.json()["id_column"]
    return id_column


def get_dataset_of_project(api_key: str, project_id: str) -> str:
    check_uuid_well_formed(project_id, "project ID")
    res = requests.get(
        cli_base_url + f"/projects/{project_id}/dataset",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    dataset_id: str = res.json()["dataset_id"]
    return dataset_id


def get_dataset_schema(api_key: str, dataset_id: str) -> JSONDict:
    check_uuid_well_formed(dataset_id, "dataset ID")
    res = requests.get(
        cli_base_url + f"/datasets/{dataset_id}/schema",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    schema: JSONDict = res.json()["schema"]
    return schema


def get_dataset_details(api_key: str, dataset_id: str, task_type: Optional[str]) -> JSONDict:
    check_uuid_well_formed(dataset_id, "dataset ID")
    res = requests.get(
        project_base_url + f"/dataset_details/{dataset_id}",
        params=dict(tasktype=task_type),
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    dataset_details: JSONDict = res.json()
    return dataset_details


def check_column_diversity(api_key: str, dataset_id: str, column_name: str) -> JSONDict:
    check_uuid_well_formed(dataset_id, "dataset ID")
    res = requests.get(
        dataset_base_url + f"/diversity/{dataset_id}/{column_name}",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    column_diversity: JSONDict = res.json()
    return column_diversity


def is_valid_multilabel_column(api_key: str, dataset_id: str, column_name: str) -> bool:
    check_uuid_well_formed(dataset_id, "dataset ID")
    res = requests.get(
        dataset_base_url + f"/check_valid_multilabel/{dataset_id}/{column_name}",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    multilabel_column: JSONDict = res.json()
    return bool(multilabel_column["is_valid_multilabel_column"])


def clean_dataset(
    api_key: str,
    dataset_id: str,
    project_name: str,
    task_type: Optional[str],
    modality: str,
    model_type: str,
    label_column: Optional[str],
    feature_columns: List[str],
    text_column: Optional[str],
) -> str:
    check_uuid_well_formed(dataset_id, "dataset ID")
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
    check_uuid_well_formed(project_id, "project ID")
    res = requests.get(
        cleanset_base_url + f"/project/{project_id}/latest_cleanset_id",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    cleanset_id = res.json()["cleanset_id"]
    return str(cleanset_id)


def poll_dataset_id_for_name(api_key: str, dataset_name: str, timeout: Optional[int]) -> str:
    start_time = time.time()
    while timeout is None or time.time() - start_time < timeout:
        dataset_id = get_dataset_id_for_name(api_key, dataset_name, timeout)

        if dataset_id is not None:
            return dataset_id

        time.sleep(5)

    raise TimeoutError(f"Timed out waiting for dataset {dataset_name} to be created.")


def get_dataset_id_for_name(
    api_key: str, dataset_name: str, timeout: Optional[int]
) -> Optional[str]:
    res = requests.get(
        dataset_base_url + f"/dataset_id_for_name",
        params=dict(dataset_name=dataset_name),
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    return cast(Optional[str], res.json().get("dataset_id", None))


def get_cleanset_status(api_key: str, cleanset_id: str) -> JSONDict:
    check_uuid_well_formed(cleanset_id, "cleanset ID")
    res = requests.get(
        cleanset_base_url + f"/{cleanset_id}/status",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    status: JSONDict = res.json()
    return status


def delete_dataset(api_key: str, dataset_id: str) -> None:
    check_uuid_well_formed(dataset_id, "dataset ID")
    res = requests.delete(dataset_base_url + f"/{dataset_id}", headers=_construct_headers(api_key))
    handle_api_error(res)


def delete_project(api_key: str, project_id: str) -> None:
    check_uuid_well_formed(project_id, "project ID")
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


def poll_ingestion_progress(api_key: str, upload_id: str, description: str) -> str:
    """Polls for ingestion progress until complete, returns dataset ID."""
    check_uuid_well_formed(upload_id, "upload ID")

    with tqdm(total=1, desc=description, bar_format="{desc}: {percentage:3.0f}%|{bar}|") as pbar:
        done = False
        while not done:
            progress = get_ingestion_status(api_key, upload_id)
            status = progress.get("status")
            done = status == "complete"

            if status == "error":
                raise IngestionError(
                    progress.get("error_type", "Internal Server Error"),
                    progress.get(
                        "error_message", "Please try again or contact support@cleanlab.ai"
                    ),
                )

            # convert progress to float
            pbar.update(float(progress.get("progress", 0)) - pbar.n)
            time.sleep(1)

        # mark progress as done
        pbar.update(float(1) - pbar.n)

    # get dataset ID
    dataset_id = get_dataset_id(api_key, upload_id)["dataset_id"]
    return str(dataset_id)


def deploy_model(api_key: str, cleanset_id: str, model_name: str) -> str:
    """Deploys model and returns model ID."""
    check_uuid_well_formed(cleanset_id, "cleanset ID")
    res = requests.post(
        model_base_url,
        headers=_construct_headers(api_key),
        json=dict(cleanset_id=cleanset_id, deployment_name=model_name),
    )

    handle_api_error(res)
    model_id: str = res.json()["id"]
    return model_id


def get_deployment_status(api_key: str, model_id: str) -> str:
    """Gets status of model deployment."""
    check_uuid_well_formed(model_id, "model ID")
    res = requests.get(
        f"{model_base_url}/{model_id}",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    deployment: JSONDict = res.json()
    return str(deployment["status"])


def upload_predict_batch(api_key: str, model_id: str, batch: io.StringIO) -> str:
    """Uploads prediction batch and returns query ID."""
    check_uuid_well_formed(model_id, "model ID")
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
    check_uuid_well_formed(model_id, "model ID")
    check_uuid_well_formed(query_id, "query ID")
    res = requests.post(
        f"{model_base_url}/{model_id}/predict/{query_id}",
        headers=_construct_headers(api_key),
    )

    handle_api_error(res)


def get_prediction_status(api_key: str, query_id: str) -> Dict[str, str]:
    """Gets status of model prediction query. Returns status, and optionally the result url or error message."""
    check_uuid_well_formed(query_id, "query ID")
    res = requests.get(
        f"{model_base_url}/predict/{query_id}",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)

    return cast(Dict[str, str], res.json())


def get_deployed_model_info(api_key: str, model_id: str) -> Dict[str, str]:
    """Get info about deployed model, including model id, name, cleanset id, dataset id, projectid, updated_at, status, and tasktype"""
    check_uuid_well_formed(model_id, "model ID")
    res = requests.get(
        f"{model_base_url}/{model_id}",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)

    return cast(Dict[str, str], res.json())


def create_enrichment_project(
    api_key: str,
    dataset_id: str,
    name: str,
) -> JSONDict:
    """Create a new enrichment project."""
    check_uuid_well_formed(dataset_id, "dataset ID")
    request_json = dict(
        dataset_id=dataset_id,
        name=name,
    )
    res = requests.post(
        f"{enrichment_base_url}/projects",
        headers=_construct_headers(api_key),
        json=request_json,
    )
    handle_api_error(res)
    return cast(JSONDict, res.json())


def delete_enrichment_project(api_key: str, project_id: str) -> None:
    """Delete an enrichment project."""
    check_uuid_well_formed(project_id, "Enrichment Project ID")
    res = requests.delete(
        f"{enrichment_base_url}/projects/{project_id}", headers=_construct_headers(api_key)
    )
    handle_api_error(res)


def get_enrichment_project(api_key: str, project_id: str) -> JSONDict:
    """Get an existing enrichment project."""
    check_uuid_well_formed(project_id, "Enrichment Project ID")
    res = requests.get(
        f"{enrichment_base_url}/projects/{project_id}",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    return cast(JSONDict, res.json())


def list_all_enrichment_projects(api_key: str) -> List[JSONDict]:
    """Get a list of all enrichment projects."""
    all_projects: List[JSONDict] = []
    page: Optional[int] = 1

    while page is not None:
        res = requests.get(
            f"{enrichment_base_url}/projects?page={page}",
            headers=_construct_headers(api_key),
        )
        handle_api_error(res)
        res_json: JSONDict = res.json()

        if "projects" in res_json:
            all_projects.extend(res_json["projects"])

        page = res_json.get("next_page", None)

    return all_projects


def enrichment_preview(
    api_key: str,
    new_column_name: str,
    project_id: str,
    prompt: str,
    constrain_outputs: Optional[List[str]] = None,
    extraction_pattern: Optional[str] = None,
    indices: Optional[List[int]] = None,
    optimize_prompt: Optional[bool] = True,
    quality_preset: Optional[TLMQualityPreset] = "medium",
    replacements: Optional[List[Dict[str, str]]] = [],
    tlm_options: Optional[Dict[str, Any]] = {},
) -> JSONDict:
    """Call Enrichment Preview API and get response."""
    check_uuid_well_formed(project_id, "project_id")
    request_json = dict(
        new_column_name=new_column_name,
        project_id=project_id,
        prompt=prompt,
        constrain_outputs=constrain_outputs,
        extraction_pattern=extraction_pattern,
        indices=indices,
        optimize_prompt=optimize_prompt,
        replacements=replacements,
        tlm_options=tlm_options,
        tlm_quality_preset=quality_preset,
    )

    res = requests.post(
        f"{enrichment_base_url}/preview",
        headers=_construct_headers(api_key),
        json=request_json,
    )
    handle_api_error(res)
    return cast(JSONDict, res.json())


def enrichment_run(
    api_key: str,
    new_column_name: str,
    project_id: str,
    prompt: str,
    constrain_outputs: Optional[List[str]] = None,
    extraction_pattern: Optional[str] = None,
    optimize_prompt: Optional[bool] = True,
    quality_preset: Optional[TLMQualityPreset] = "medium",
    replacements: Optional[List[Dict[str, str]]] = [],
    tlm_options: Optional[Dict[str, Any]] = {},
) -> JSONDict:
    """Call Enrichment Enrich_all API and get response."""
    check_uuid_well_formed(project_id, "project_id")
    request_json = dict(
        new_column_name=new_column_name,
        project_id=project_id,
        prompt=prompt,
        constrain_outputs=constrain_outputs,
        extraction_pattern=extraction_pattern,
        optimize_prompt=optimize_prompt,
        replacements=replacements,
        tlm_options=tlm_options,
        tlm_quality_preset=quality_preset,
    )

    res = requests.post(
        f"{enrichment_base_url}/enrich_all",
        headers=_construct_headers(api_key),
        json=request_json,
    )
    handle_api_error(res)
    return cast(JSONDict, res.json())


def get_enrichment_job_status(api_key: str, job_id: str) -> JSONDict:
    """Get status of enrichment job."""
    check_uuid_well_formed(job_id, "job_id")

    res = requests.get(
        f"{enrichment_base_url}/status/{job_id}",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    return cast(JSONDict, res.json())


def get_enrichment_job_result(
    api_key: str, job_id: str, page: int, include_original_dataset: Optional[bool] = False
) -> List[JSONDict]:
    """Get result of enrichment job.

    Args:
        api_key (str): studio API key for auth
        job_id (str): job id
        page (int): page number
        include_original_dataset (bool): whether to return only results or merged results and original dataset directly from the backend
    """
    check_uuid_well_formed(job_id, "job_id")

    res = requests.get(
        f"{enrichment_base_url}/enrich_all/{job_id}",
        headers=_construct_headers(api_key),
        params=dict(page=page, include_original_dataset=include_original_dataset),
    )
    handle_api_error(res)
    return cast(List[JSONDict], res.json())


def list_enrichment_jobs(api_key: str, project_id: str) -> List[JSONDict]:
    """List all enrichment jobs for a project."""
    check_uuid_well_formed(project_id, "project_id")
    res = requests.get(
        f"{enrichment_base_url}/projects/{project_id}/jobs",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    return cast(List[JSONDict], res.json())


def get_enrichment_job(api_key: str, job_id: str) -> JSONDict:
    """Get enrichment job."""
    check_uuid_well_formed(job_id, "job_id")

    res = requests.get(
        f"{enrichment_base_url}/jobs/{job_id}",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    return cast(JSONDict, res.json())


def export_results(api_key: str, job_id: str, filename: Optional[str] = None) -> str:
    """
    Exports the results of a job to a CSV file.

    Args:
        api_key (str): The API key used for authentication.
        job_id (str): The unique identifier of the job whose results are to be exported.
        filename (str): The name of the CSV file to save the results to. If None, a default filename is generated.

    Returns:
        str: A message indicating the CSV file has been saved, including the filename.
    """
    check_uuid_well_formed(job_id, "job_id")
    if filename is None:
        filename = f"enrichment_results_{job_id}.csv"
    check_valid_csv_filename(filename)
    res = requests.get(
        f"{enrichment_base_url}/export/{job_id}",
        headers=_construct_headers(api_key),
    )
    if res.status_code == 200:
        with open(filename, "wb") as file:
            file.write(res.content)
    else:
        handle_api_error(res)
    return filename


def pause_enrichment_job(api_key: str, job_id: str) -> None:
    """Pause enrichment job."""
    check_uuid_well_formed(job_id, "job_id")

    res = requests.post(
        f"{enrichment_base_url}/enrich_all/{job_id}/pause",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)


def resume_enrichment_job(api_key: str, job_id: str) -> JSONDict:
    """Resume enrichment job."""
    check_uuid_well_formed(job_id, "job_id")

    res = requests.post(
        f"{enrichment_base_url}/enrich_all/{job_id}/resume",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    return cast(JSONDict, res.json())


def tlm_retry(func: Callable[..., Any]) -> Callable[..., Any]:
    """Implements TLM retry decorator, with special handling for rate limit retries."""

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # total number of tries = number of retries + original try
        max_general_retries = kwargs.pop("retries", 0)
        max_connection_error_retries = 20

        sleep_time = 0
        error_message = ""

        num_general_retry = 0
        num_connection_error_retry = 0

        while (
            num_general_retry <= max_general_retries
            and num_connection_error_retry <= max_connection_error_retries
        ):
            await asyncio.sleep(sleep_time)
            try:
                return await func(*args, **kwargs)
            except ssl.SSLCertVerificationError as e:
                warnings.warn(
                    "Please ensure that your SSL certificates are up to date. If you installed python via python pkg installer, please make sure to execute 'Install Certificates.command' in the python installation directory."
                )
                raise
            except aiohttp.client_exceptions.ClientConnectorError as e:
                if num_connection_error_retry == (max_connection_error_retries // 2):
                    warnings.warn(
                        f"Connection error after {num_connection_error_retry} retries. Retrying..."
                    )
                sleep_time = min(2**num_connection_error_retry, 60)
                # note: we have a different counter for connection errors, because we want to retry connection errors more times
                num_connection_error_retry += 1
                error_message = str(e)
            except RateLimitError as e:
                # note: we don't increment num_general_retry here, because we don't want rate limit retries to count against the total number of retries
                sleep_time = e.retry_after
            except TlmBadRequest as e:
                # dont retry for client-side errors
                raise e
            except Exception as e:
                sleep_time = 2**num_general_retry
                num_general_retry += 1
                error_message = str(e)

        if num_connection_error_retry > max_connection_error_retries:
            raise APIError(
                f"Connection error after {num_connection_error_retry} retries. {error_message}",
                -1,
            )
        else:
            raise APIError(f"TLM failed after {num_general_retry} attempts. {error_message}", -1)

    return wrapper


@tlm_retry
async def tlm_prompt(
    api_key: str,
    prompt: str,
    quality_preset: str,
    options: Optional[JSONDict],
    rate_handler: TlmRateHandler,
    client_session: Optional[aiohttp.ClientSession] = None,
    batch_index: Optional[int] = None,
    constrain_outputs: Optional[List[str]] = None,
) -> JSONDict:
    """
    Prompt Trustworthy Language Model with a question, and get back its answer along with a confidence score

    Args:
        api_key (str): studio API key for auth
        prompt (str): prompt for TLM to respond to
        quality_preset (str): quality preset to use to generate response
        options (JSONDict): additional parameters for TLM
        rate_handler (TlmRateHandler): concurrency handler used to manage TLM request rate
        client_session (aiohttp.ClientSession): client session used to issue TLM request
        batch_index (Optional[int], optional): index of prompt in batch, used for error messages. Defaults to None if not in batch.
        constrain_outputs (Optional[List[str]], optional): list of strings to constrain the output of the TLM to. Defaults to None.
    Returns:
        JSONDict: dictionary with TLM response and confidence score
    """
    local_scoped_client = False
    if not client_session:
        client_session = aiohttp.ClientSession()
        local_scoped_client = True

    try:
        async with rate_handler:
            base_api_url = os.environ.get("CLEANLAB_API_TLM_BASE_URL", tlm_base_url)
            res = await client_session.post(
                f"{base_api_url}/prompt",
                json=dict(
                    prompt=prompt,
                    quality=quality_preset,
                    options=options or {},
                    user_id=api_key,
                    client_id=api_key,
                    constrain_outputs=constrain_outputs,
                ),
                headers=_construct_headers(api_key),
            )

            res_json = await res.json()

            handle_rate_limit_error_from_resp(res)
            await handle_tlm_client_error_from_resp(res, batch_index)
            await handle_tlm_api_error_from_resp(res, batch_index)

            if not res_json.get("deberta_success", True):
                raise TlmPartialSuccess("Partial failure on deberta call -- slowdown request rate.")

    finally:
        if local_scoped_client:
            await client_session.close()

    return cast(JSONDict, res_json)


@tlm_retry
async def tlm_get_confidence_score(
    api_key: str,
    prompt: str,
    response: str,
    quality_preset: str,
    options: Optional[JSONDict],
    rate_handler: TlmRateHandler,
    client_session: Optional[aiohttp.ClientSession] = None,
    batch_index: Optional[int] = None,
) -> JSONDict:
    """
    Query Trustworthy Language Model for a confidence score for the prompt-response pair.

    Args:
        api_key (str): studio API key for auth
        prompt (str): prompt for TLM to get confidence score for
        response (str): response for TLM to get confidence score for
        quality_preset (str): quality preset to use to generate confidence score
        options (JSONDict): additional parameters for TLM
        rate_handler (TlmRateHandler): concurrency handler used to manage TLM request rate
        client_session (aiohttp.ClientSession): client session used to issue TLM request
        batch_index (Optional[int], optional): index of prompt in batch, used for error messages. Defaults to None if not in batch.

    Returns:
        JSONDict: dictionary with TLM confidence score
    """
    local_scoped_client = False
    if not client_session:
        client_session = aiohttp.ClientSession()
        local_scoped_client = True

    try:
        async with rate_handler:
            res = await client_session.post(
                f"{tlm_base_url}/get_confidence_score",
                json=dict(
                    prompt=prompt,
                    response=response,
                    quality=quality_preset,
                    options=options or {},
                ),
                headers=_construct_headers(api_key),
            )

            res_json = await res.json()

            handle_rate_limit_error_from_resp(res)
            await handle_tlm_client_error_from_resp(res, batch_index)
            await handle_tlm_api_error_from_resp(res, batch_index)

    finally:
        if local_scoped_client:
            await client_session.close()

    return cast(JSONDict, res_json)


def send_telemetry(info: JSONDict) -> None:
    requests.post(f"{cli_base_url}/telemetry", json=info)
