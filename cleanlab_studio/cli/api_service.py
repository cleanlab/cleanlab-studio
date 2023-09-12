"""
Methods for interacting with the command line server API
1:1 mapping with command_line_api.py
"""
import gzip
import json
import os
import pathlib
import asyncio
from typing import List, Any, Optional, Tuple

import aiohttp
import requests
import pandas as pd

from cleanlab_studio.version import __version__
from cleanlab_studio.cli.click_helpers import abort, warn, info
from cleanlab_studio.cli.dataset.image_utils import get_image_filepath
from cleanlab_studio.cli.dataset.schema_types import Schema
from cleanlab_studio.cli.types import JSONDict, IDType, MEDIA_MODALITIES

base_url = os.environ.get("CLEANLAB_API_BASE_URL", "https://api.cleanlab.ai/api")
cli_base_url = f"{base_url}/cli/v0"


MAX_PARALLEL_UPLOADS = 32  # XXX choose this dynamically?
INITIAL_BACKOFF = 0.25  # seconds
MAX_RETRIES = 4


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


def initialize_dataset(api_key: str, schema: Schema) -> str:
    request_json = dict(schema=schema.to_dict())
    res = requests.post(
        cli_base_url + "/datasets", json=request_json, headers=_construct_headers(api_key)
    )
    handle_api_error(res)
    dataset_id: str = res.json()["dataset_id"]
    return dataset_id


def get_existing_ids(api_key: str, dataset_id: str) -> List[IDType]:
    res = requests.get(
        cli_base_url + f"/datasets/{dataset_id}/ids",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    existing_ids: List[IDType] = res.json()["existing_ids"]
    return existing_ids


def get_dataset_schema(api_key: str, dataset_id: str) -> Schema:
    res = requests.get(
        cli_base_url + f"/datasets/{dataset_id}/schema",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    schema = Schema.from_dict(res.json()["schema"])
    return schema


async def upload_rows_async(
    session: aiohttp.ClientSession,
    api_key: str,
    dataset_id: str,
    dataset_filepath: str,
    schema: Schema,
    rows: List[Any],
    filepath_columns: List[str],
    upload_sem: asyncio.Semaphore,
) -> None:
    """
    Upload rows of dataset

    :param session: client session for making http requests
    :param api_key: api key for getting presigned posts
    :param dataset_id: id of dataset to upload files for
    :param dataset_filepath: filepath of dataset to upload files for (can be empty string for simple image upload)
    :param schema: schema of dataset to upload
    :param rows: rows of dataset to upload
    :param filepath_columns: names of any columns containing paths of files that should be uploaded to S3
    :param upload_sem: sempahore to bound number of parallel uploadsb
    """
    modality = schema.metadata.modality
    assert len(filepath_columns) == 0 or modality in MEDIA_MODALITIES
    needs_media_upload = modality in MEDIA_MODALITIES and filepath_columns
    columns = list(schema.fields.keys())

    async with upload_sem:
        if needs_media_upload:
            id_column = schema.metadata.id_column
            assert id_column is not None
            id_column_idx = columns.index(id_column)
            row_ids = [row[id_column_idx] for row in rows]

            upload_tasks = [
                upload_files_for_filepath_column(
                    session=session,
                    api_key=api_key,
                    dataset_id=dataset_id,
                    dataset_filepath=dataset_filepath,
                    rows=rows,
                    row_ids=row_ids,
                    columns=columns,
                    filepath_column=col,
                    media_type=modality.value,
                )
                for col in filepath_columns
            ]
            await asyncio.gather(*upload_tasks)

        url = cli_base_url + f"/datasets/{dataset_id}"
        data = gzip.compress(
            json.dumps(dict(rows=json.dumps(rows), columns=json.dumps(columns))).encode("utf-8")
        )
        headers = _construct_headers(api_key)
        headers["Content-Encoding"] = "gzip"

        async with session.post(url=url, data=data, headers=headers) as res:
            res_text = await res.read()
            handle_api_error_from_json(json.loads(res_text))


async def upload_files_for_filepath_column(
    session: aiohttp.ClientSession,
    api_key: str,
    dataset_id: str,
    dataset_filepath: str,
    rows: List[List[Any]],
    row_ids: List[Any],
    columns: List[str],
    filepath_column: str,
    media_type: str,
) -> None:
    """
    Uploads all files in a filepath column to S3

    :param session: client session for making http requests
    :param api_key: api key for getting presigned posts
    :param dataset_id: id of dataset to upload files for
    :param dataset_filepath: filepath of dataset to upload files for (can be empty string for simple image upload)
    :param rows: dataset rows
    :param row_ids: ids of dataset rows
    :param columns: dataset column names
    :param filepath_column: name of the column to upload files for
    :param media_type: type of media to upload
    """
    sem = asyncio.Semaphore(MAX_PARALLEL_UPLOADS)
    cancelled = asyncio.Event()
    filepath_column_idx = columns.index(filepath_column)
    filepaths: List[str] = [row[filepath_column_idx] for row in rows]
    dataset_dir: pathlib.Path = pathlib.Path(dataset_filepath).parent
    absolute_filepaths = [
        str(get_image_filepath(dataset_dir, row[filepath_column_idx])) for row in rows
    ]
    filepath_to_posts = get_presigned_posts(
        api_key=api_key,
        dataset_id=dataset_id,
        filepaths=filepaths,
        row_ids=row_ids,
        media_type=media_type,
    )
    for coro in asyncio.as_completed(
        [
            post_file(
                session,
                original_filepath,
                absolute_filepath,
                dict(filepath_to_posts[original_filepath]),
                sem,
                cancelled,
            )
            for original_filepath, absolute_filepath in zip(filepaths, absolute_filepaths)
        ]
    ):
        ok, original_filepath = await coro
        if ok:
            info(f"Uploaded {original_filepath}")
        else:
            # cancel tasks we don't need anymore, to give us flexibility to
            # not abort() later but to throw an exception and keep the
            # Python interpreter running without a bunch of leaked
            # coroutines
            cancelled.set()
            abort(f"Failed to upload {original_filepath}")


async def post_file(
    session: aiohttp.ClientSession,
    original_filepath: str,
    absolute_filepath: str,
    post_data: JSONDict,
    sem: asyncio.Semaphore,
    cancelled: asyncio.Event,
) -> Tuple[Optional[bool], str]:
    """
    Upload a single file using a presigned post

    :param session: client session for making http requests
    :original_filepath: the original filepath that appears in the dataset
    :absolute_filepath: the absolute path to the file
    :presigned_post: presigned post to use to upload the file to S3
    :sem: semaphore for parallel uploads
    :cancelled: event that signals if upload should be cancelled
    """
    async with sem:
        presigned_post = post_data["post"]
        # note: we pass in the path and we don't open the file until we
        # get in here, so we don't have tons of concurrently opened
        # files
        if not cancelled.is_set():
            retries = MAX_RETRIES
            wait = INITIAL_BACKOFF
            while retries:
                try:
                    async with session.post(
                        url=presigned_post["url"],
                        data={
                            **presigned_post["fields"],
                            "file": open(absolute_filepath, "rb"),
                        },
                    ) as res:
                        if res.ok:
                            return res.ok, original_filepath
                except Exception:
                    pass  # ignore, will retry
                await asyncio.sleep(wait)
                wait *= 2
                retries -= 1
            return False, original_filepath
    return None, original_filepath


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
