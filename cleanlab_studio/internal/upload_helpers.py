import asyncio
import functools
import json
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from multidict import CIMultiDictProxy
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

try:
    import bigframes.pandas as bpd

    _bigframes_exists = True
except ImportError:
    _bigframes_exists = False

from cleanlab_studio.errors import InvalidInputError, InvalidSchemaTypeError

from .api import api
from .dataset_source import DatasetSource
from .types import JSONDict, SchemaOverride


def upload_dataset(
    api_key: str,
    dataset_source: DatasetSource,
    *,
    schema_overrides: Optional[List[SchemaOverride]] = None,
) -> str:
    # perform file upload
    upload_id = upload_dataset_file(api_key, dataset_source)

    # confirm upload (and kick off processing)
    api.confirm_upload(api_key, upload_id, schema_overrides)

    # wait for dataset upload
    dataset_id = api.poll_ingestion_progress(api_key, upload_id, "Ingesting Dataset...")

    # return dataset id
    return dataset_id


def upload_url_dataset(
    api_key: str,
    url: str,
    *,
    schema_overrides: Optional[List[SchemaOverride]] = None,
) -> str:
    # start dataset upload
    upload_id = api.start_url_upload(api_key, url, schema_overrides)

    # wait for dataset upload
    dataset_id = api.poll_ingestion_progress(api_key, upload_id, "Ingesting Dataset...")

    # return dataset id
    return dataset_id


def upload_bigquery_dataset(
    api_key: str,
    bigquery_project: str,
    bigquery_dataset_id: str,
    bigquery_table_id: str,
    *,
    schema_overrides: Optional[List[SchemaOverride]] = None,
    bqclient: Optional[Any] = None,
) -> str:
    # add bigquery data viewer role for service account
    _add_bq_data_viewer_role(
        api_key, f"{bigquery_project}.{bigquery_dataset_id}.{bigquery_table_id}", bqclient=bqclient
    )

    # start dataset upload
    upload_id = api.start_bigquery_upload(
        api_key, bigquery_project, bigquery_dataset_id, bigquery_table_id, schema_overrides
    )

    # wait for dataset upload
    dataset_id = api.poll_ingestion_progress(api_key, upload_id, "Ingesting Dataset...")

    # return dataset id
    return dataset_id


def upload_bigframe_dataset(
    api_key: str,
    bigframe: Any,
    *,
    schema_overrides: Optional[List[SchemaOverride]] = None,
) -> str:
    if not _bigframes_exists:
        raise ImportError(
            "BigFrames is not installed. Please install BigFrames to use this feature."
        )

    if not isinstance(bigframe, bpd.DataFrame):
        raise InvalidInputError("Expected bigframe to be a BigFrames DataFrame.")

    table_id = bigframe.to_gbq()
    bqclient = bigframe.bqclient

    bq_project_id, bq_dataset_id, bq_table_id = table_id.split(".")
    return upload_bigquery_dataset(
        api_key,
        bq_project_id,
        bq_dataset_id,
        bq_table_id,
        schema_overrides=schema_overrides,
        bqclient=bqclient,
    )


def _add_bq_data_viewer_role(api_key: str, table_id: str, bqclient: Optional[Any] = None) -> None:
    if bqclient is None:
        from google.cloud import bigquery

        bqclient = bigquery.Client()

    bigquery_principal_account = api.get_bigquery_principal_account(api_key)
    policy = bqclient.get_iam_policy(table_id)

    binding = {
        "role": "roles/bigquery.dataViewer",
        "members": {f"serviceAccount:{bigquery_principal_account}"},
    }

    policy.bindings.append(binding)
    bqclient.set_iam_policy(table_id, policy)


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


def convert_schema_overrides(schema_overrides: Dict[str, Dict[str, Any]]) -> List[SchemaOverride]:
    return [
        SchemaOverride(
            name=col,
            column_type=_old_schema_to_column_type(
                old_schema["data_type"], old_schema["feature_type"]
            ),
        )
        for col, old_schema in schema_overrides.items()
    ]


def _old_schema_to_column_type(data_type: str, feature_type: str) -> str:
    if data_type == "string":
        return _string_data_type_to_column_type(data_type, feature_type)
    if data_type == "integer":
        return _integer_data_type_to_column_type(data_type, feature_type)
    if data_type == "float":
        return _float_data_type_to_column_type(data_type, feature_type)
    if data_type == "boolean":
        return _boolean_data_type_to_column_type(data_type, feature_type)
    raise InvalidSchemaTypeError(f"Unsupported data type: {data_type}.")


def _string_data_type_to_column_type(data_type: str, feature_type: str) -> str:
    if feature_type in ["text", "categorical"]:
        return "string"
    if feature_type == "image":
        return "image_external"
    if feature_type in ["datetime", "identifier"]:
        raise InvalidSchemaTypeError(
            f"Cannot convert old schema feature type '{feature_type}' to new schema type."
        )
    raise InvalidSchemaTypeError(
        f"Unsupported data type, feature type combination: {data_type}, {feature_type}."
    )


def _integer_data_type_to_column_type(data_type: str, feature_type: str) -> str:
    if feature_type in ["categorical", "numeric"]:
        return "integer"
    if feature_type in ["datetime", "identifier"]:
        raise InvalidSchemaTypeError(
            f"Cannot convert old schema feature type '{feature_type}' to new schema type."
        )
    raise InvalidSchemaTypeError(
        f"Unsupported data type, feature type combination: {data_type}, {feature_type}."
    )


def _float_data_type_to_column_type(data_type: str, feature_type: str) -> str:
    if feature_type == "numeric":
        return "float"
    if feature_type == "datetime":
        raise InvalidSchemaTypeError(
            f"Cannot convert old schema feature type '{feature_type}' to new schema type."
        )
    raise InvalidSchemaTypeError(
        f"Unsupported data type, feature type combination: {data_type}, {feature_type}."
    )


def _boolean_data_type_to_column_type(data_type: str, feature_type: str) -> str:
    if feature_type == "boolean":
        return "boolean"
    raise InvalidSchemaTypeError(
        f"Unsupported data type, feature type combination: {data_type}, {feature_type}."
    )
