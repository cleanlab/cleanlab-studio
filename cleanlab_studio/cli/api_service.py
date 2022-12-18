"""
Methods for interacting with the command line server API
1:1 mapping with command_line_api.py
"""
import gzip
import json
import os
import asyncio
from typing import List, Any, Optional, Tuple

import aiohttp
import requests

from cleanlab_studio.version import __version__
from cleanlab_studio.cli.click_helpers import abort, warn, info
from cleanlab_studio.cli.dataset.image_utils import get_image_filepath
from cleanlab_studio.internal.schema import Schema
from cleanlab_studio.internal.types import JSONDict, IDType, Modality
from cleanlab_studio.internal.api import base_url, _construct_headers, handle_api_error, handle_api_error_from_json, get_presigned_posts



MAX_PARALLEL_UPLOADS = 32  # XXX choose this dynamically?
INITIAL_BACKOFF = 0.25  # seconds
MAX_RETRIES = 4




async def upload_rows_async(
    session: aiohttp.ClientSession,
    api_key: str,
    dataset_id: str,
    dataset_filepath: str,
    schema: Schema,
    rows: List[Any],
) -> None:
    modality = schema.metadata.modality
    needs_media_upload = modality in [Modality.image]
    columns = list(schema.fields.keys())

    if needs_media_upload:
        id_column = schema.metadata.id_column
        assert id_column is not None
        id_column_idx = columns.index(id_column)
        row_ids = [row[id_column_idx] for row in rows]

        filepath_column = schema.metadata.filepath_column
        assert filepath_column is not None
        filepath_column_idx = columns.index(filepath_column)
        filepaths = [row[filepath_column_idx] for row in rows]
        absolute_filepaths = [
            get_image_filepath(row[filepath_column_idx], dataset_filepath) for row in rows
        ]

        filepath_to_post = get_presigned_posts(
            api_key=api_key,
            dataset_id=dataset_id,
            filepaths=filepaths,
            row_ids=row_ids,
            media_type=modality.value,
        )
        sem = asyncio.Semaphore(MAX_PARALLEL_UPLOADS)
        cancelled = False

        async def post_file(
            original_filepath: str, absolute_filepath: str
        ) -> Tuple[Optional[bool], str]:
            async with sem:
                # note: we pass in the path and we don't open the file until we
                # get in here, so we don't have tons of concurrently opened
                # files
                if not cancelled:
                    post_data = filepath_to_post.get(original_filepath)
                    assert isinstance(post_data, dict)
                    presigned_post = post_data["post"]

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

        for coro in asyncio.as_completed(
            [
                post_file(original_filepath, absolute_filepath)
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
                cancelled = True
                abort(f"Failed to upload {original_filepath}")

    url = base_url + f"/datasets/{dataset_id}"
    data = gzip.compress(
        json.dumps(dict(rows=json.dumps(rows), columns=json.dumps(columns))).encode("utf-8")
    )
    headers = _construct_headers(api_key)
    headers["Content-Encoding"] = "gzip"

    async with session.post(url=url, data=data, headers=headers) as res:
        res_text = await res.read()
        handle_api_error_from_json(json.loads(res_text))


def get_completion_status(api_key: str, dataset_id: str) -> bool:
    res = requests.get(
        base_url + f"/datasets/{dataset_id}/complete",
        headers=_construct_headers(api_key),
    )
    handle_api_error(res)
    completed: bool = res.json()["complete"]
    return completed


def get_id_column(api_key: str, cleanset_id: str) -> str:
    res = requests.get(
        base_url + f"/cleansets/{cleanset_id}/id_column",
        headers={"Authorization": f"bearer {api_key}"},
    )
    handle_api_error(res)
    id_column: str = res.json()["id_column"]
    return id_column


def check_dataset_limit(
    api_key: str, file_size: int, image_dataset: bool = False, show_warning: bool = False
) -> JSONDict:
    res = requests.post(
        base_url + "/check_dataset_limit",
        json=dict(file_size=file_size, image_dataset=image_dataset),
        headers=_construct_headers(api_key),
    )
    handle_api_error(res, show_warning=show_warning)
    res_json: JSONDict = res.json()
    return res_json
