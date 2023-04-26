from typing import Optional
from cleanlab_studio.internal.api import api
from cleanlab_studio.internal.dataset_source import DatasetSource
from cleanlab_studio.internal.schema import Schema
from cleanlab_studio.internal.upload_helpers import (
    get_ingestion_result,
    get_proposed_schema,
    upload_dataset_file,
)


def upload_tabular_dataset(api_key: str, dataset_source: DatasetSource, schema: Optional[Schema]):
    upload_id = upload_dataset_file(api_key, dataset_source)
    if schema is not None:
        schema.validate()
    else:
        schema = get_proposed_schema(api_key, upload_id)
        schema.metadata.name = dataset_source.dataset_name
    api.confirm_schema(api_key, schema, upload_id)
    dataset_id = get_ingestion_result(api_key, upload_id)
    return dataset_id
