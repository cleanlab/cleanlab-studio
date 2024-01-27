import asyncio
import functools
import json
from typing import List, Optional, Any
from tqdm import tqdm, trange

import aiohttp
from multidict import CIMultiDictProxy
import requests
from requests.adapters import HTTPAdapter, Retry

from .api import api
from .dataset_source import DatasetSource, LocalDatasetSource
from .types import FieldSchemaDict, JSONDict

_snowflake_exists = api.snowflake_exists
_pyspark_exists = api.pyspark_exists
_lazy_loaded_dataset_source_exists = _snowflake_exists or _pyspark_exists
if _lazy_loaded_dataset_source_exists:
    from .dataset_source import LazyLoadedDatasetSource


def upload_dataset(
    api_key: str,
    dataset_source: DatasetSource,
    *,
    schema_overrides: Optional[FieldSchemaDict] = None,
    modality: Optional[str] = None,
    id_column: Optional[str] = None,
) -> str:
    upload_id = upload_dataset_file(api_key, dataset_source)
    schema = get_proposed_schema(api_key, upload_id)

    if (schema is None or schema.get("immutable", False)) and (
        schema_overrides is not None or modality is not None or id_column is not None
    ):
        raise ValueError(
            "Schema_overrides, modality, and id_column parameters cannot be provided for simple zip uploads"
        )

    if schema is not None and not schema.get("immutable", False):
        schema["metadata"]["name"] = dataset_source.dataset_name
        if schema_overrides is not None:
            for field in schema_overrides:
                schema["fields"][field] = schema_overrides[field]
        if modality is not None:
            schema["metadata"]["modality"] = modality
        if id_column is not None:
            if id_column not in schema["fields"]:
                raise ValueError(
                    f"ID column {id_column} not found in dataset columns: {list(schema['fields'].keys())}"
                )
            schema["metadata"]["id_column"] = id_column

    api.confirm_schema(api_key, schema, upload_id)
    dataset_id = get_ingestion_result(api_key, upload_id)
    return dataset_id


async def _upload_file_chunk_async(
    session: aiohttp.ClientSession,
    chunk: bytes,
    presigned_post: str,
) -> CIMultiDictProxy[str]:
    async with session.put(presigned_post, data=chunk) as response:
        return response.headers


async def upload_file_parts_async(
    dataset_source: LocalDatasetSource, part_sizes: List[int], presigned_posts: List[str]
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
    dataset_source: LocalDatasetSource, part_sizes: List[int], presigned_posts: List[str]
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


def upload_stream_parts(
    api_key: str, upload_id: str, dataset_source: Any, part_size: int
) -> List[JSONDict]:
    responses = []
    part_number = 1

    t = trange(
        100, desc="Uploading dataset...", leave=True, bar_format="{desc}: {percentage:3.0f}%|{bar}|"
    )

    for chunk, rows in dataset_source.get_chunks(chunk_size=part_size):
        resp = api.upload_stream_part(api_key, upload_id, part_number, chunk)

        responses.append(resp["ETag"])
        part_number += 1

        t.update(rows // dataset_source.total_rows * 100)

    t.close()

    return [{"ETag": json.loads(etag), "PartNumber": i + 1} for i, etag in enumerate(responses)]


def upload_dataset_file(api_key: str, dataset_source: DatasetSource) -> str:
    if _lazy_loaded_dataset_source_exists and isinstance(dataset_source, LazyLoadedDatasetSource):
        upload_id, part_size = api.initialize_stream_upload(
            api_key, dataset_source.get_filename(), dataset_source.get_file_type()
        )
        upload_parts = upload_stream_parts(api_key, upload_id, dataset_source, part_size)
    else:
        assert isinstance(dataset_source, LocalDatasetSource)
        upload_id, part_sizes, presigned_posts = api.initialize_upload(
            api_key,
            dataset_source.get_filename(),
            dataset_source.get_file_type(),
            dataset_source.file_size,
        )
        upload_parts = upload_file_parts(dataset_source, part_sizes, presigned_posts)
    api.complete_file_upload(api_key, upload_id, upload_parts)
    return upload_id


def get_proposed_schema(api_key: str, upload_id: str) -> Optional[JSONDict]:
    res = api.poll_progress(
        upload_id,
        functools.partial(api.get_proposed_schema, api_key),
        "Generating schema...",
    )
    schema = res.get("schema")
    return schema


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
