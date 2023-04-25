import asyncio
import functools
import json
import mimetypes
import pathlib
from typing import List, Optional

import aiohttp
from multidict import CIMultiDictProxy

from .api import api
from .schema import Schema
from .types import JSONDict


def _get_file_chunks(filepath: pathlib.Path, chunk_sizes: List[int]) -> List[bytes]:
    with open(filepath, "rb") as f:
        return [f.read(chunk_size) for chunk_size in chunk_sizes]


async def _upload_file_chunk_async(
    session: aiohttp.ClientSession,
    chunk: bytes,
    presigned_post: str,
) -> CIMultiDictProxy[str]:
    async with session.put(presigned_post, data=chunk) as response:
        return response.headers


async def upload_file_parts_async(
    filepath: pathlib.Path, part_sizes: List[int], presigned_posts: List[str]
) -> List[JSONDict]:
    tasks = []
    chunks = _get_file_chunks(filepath, part_sizes)
    async with aiohttp.ClientSession() as session:
        for chunk, presigned_post in zip(chunks, presigned_posts):
            tasks.append(
                asyncio.create_task(_upload_file_chunk_async(session, chunk, presigned_post))
            )
        task_results = await asyncio.gather(*tasks)
    return [
        {"ETag": json.loads(res.get("etag")), "PartNumber": i + 1}
        for i, res in enumerate(task_results)
    ]


def upload_dataset_file(api_key: str, filepath: pathlib.Path) -> str:
    filename = filepath.name
    file_size = filepath.stat().st_size
    file_type = mimetypes.guess_type(filename)
    upload_id, part_sizes, presigned_posts = api.initialize_upload(
        api_key, filename, file_type, file_size
    )
    upload_parts = asyncio.run(upload_file_parts_async(filepath, part_sizes, presigned_posts))
    api.complete_file_upload(api_key, upload_id, upload_parts)
    return upload_id


def get_proposed_schema(api_key: str, upload_id: str) -> Optional[Schema]:
    res = api.poll_progress(
        upload_id,
        functools.partial(api.get_proposed_schema, api_key),
        "Generating schema...",
    )
    schema = res.get("schema")
    return None if schema is None else Schema.from_dict(res["schema"])


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
    return res["dataset_id"]
