from typing import Optional

from cleanlab_studio.internal.api import api
from cleanlab_studio.internal.dataset_source import DatasetSource
from cleanlab_studio.internal.types import FieldSchemaDict
from cleanlab_studio.internal.upload_helpers import (
    get_ingestion_result,
    get_proposed_schema,
    upload_dataset_file,
)


def upload_dataset(
    api_key: str,
    dataset_source: DatasetSource,
    *,
    schema_overrides: Optional[FieldSchemaDict] = None,
    modality: Optional[str] = None,
    id_column: Optional[str] = None,
):
    upload_id = upload_dataset_file(api_key, dataset_source)
    schema = get_proposed_schema(api_key, upload_id)
    if schema_overrides is not None:
        for field in schema_overrides:
            schema["fields"][field] = schema_overrides[field]
    if modality is not None:
        schema["metadata"]["modality"] = modality
    if id_column is not None:
        schema["metadata"]["id_column"] = id_column

    api.confirm_schema(api_key, schema, upload_id)
    dataset_id = get_ingestion_result(api_key, upload_id)
    return dataset_id
