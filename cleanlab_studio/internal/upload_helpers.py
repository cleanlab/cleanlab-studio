import asyncio
import functools
import json
from typing import List, Optional
from tqdm import tqdm

import aiohttp
from multidict import CIMultiDictProxy
import requests
from requests.adapters import HTTPAdapter, Retry

from .api import api
from .dataset_source import DatasetSource
from .types import FieldSchemaDict, JSONDict


def upload_dataset(
    api_key: str,
    dataset_source: DatasetSource,
    *,
    schema_overrides: Optional[FieldSchemaDict] = None,
) -> str:
    # perform file upload
    upload_id = upload_dataset_file(api_key, dataset_source)

    # confirm upload (and kick off processing)
    api.confirm_upload(api_key, upload_id, dataset_source.get_filename())

    # wait for dataset upload
    dataset_id = api.poll_ingestion_progress(api_key, upload_id, "Ingesting Dataset...")

    # if schema overrides, update schema and wait for reingestion
    if schema_overrides:
        # TODO
        ...

    # return dataset id
    return dataset_id


async def _upload_file_chunk_async(
    session: aiohttp.ClientSession,
    chunk: bytes,
    presigned_post: str,
) -> CIMultiDictProxy[str]:
    async with session.put(presigned_post, data=chunk) as response:
        return response.headers


async def upload_file_parts_async(
    dataset_source: DatasetSource, part_sizes: List[int], presigned_posts: List[str]
) -> List[JSONDict]:
    tasks = []
    chunks = dataset_source.get_chunks(part_sizes)
    async with aiohttp.ClientSession() as session:
        for chunk, presigned_post in zip(chunks, presigned_posts):
            tasks.append(
                asyncio.create_task(_upload_file_chunk_async(session, chunk, presigned_post))
            )
        task_results = await asyncio.gather(*tasks)
    return [
        # TODO: need to give handle error for missing etags
        # next line currently fails typing because res.get("etag") can be None
        {"ETag": json.loads(res.get("etag")), "PartNumber": i + 1}  # type: ignore
        for i, res in enumerate(task_results)
    ]


def upload_file_parts(
    dataset_source: DatasetSource, part_sizes: List[int], presigned_posts: List[str]
) -> List[JSONDict]:
    session = requests.Session()
    session.mount("https://", adapter=HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1)))

    responses = []
    for chunk, presigned_post in tqdm(
        list(zip(dataset_source.get_chunks(part_sizes), presigned_posts)),
        desc="Uploading dataset...",
        bar_format="{desc}: {percentage:3.0f}%|{bar}|",
    ):
        resp = session.put(
            presigned_post,
            data=chunk,
        )
        resp.raise_for_status()
        responses.append(resp)

    return [
        {"ETag": json.loads(res.headers["etag"]), "PartNumber": i + 1}
        for i, res in enumerate(responses)
    ]


def upload_dataset_file(api_key: str, dataset_source: DatasetSource) -> str:
    upload_id, part_sizes, presigned_posts = api.initialize_upload(
        api_key,
        dataset_source.get_filename(),
        dataset_source.file_type,
        dataset_source.file_size,
    )
    upload_parts = upload_file_parts(dataset_source, part_sizes, presigned_posts)
    api.complete_file_upload(api_key, upload_id, upload_parts)
    return upload_id


def get_ingestion_result(
    api_key: str,
    upload_id: str,
) -> str:
    api.poll_progress(
        upload_id,
        functools.partial(api.get_ingestion_status, api_key),
        "Ingesting Dataset...",
    )
    res = api.get_dataset_id(api_key, upload_id)
    return str(res["dataset_id"])
