from typing import Optional

from . import upload
from cleanlab_studio.internal.api import api
from cleanlab_studio.internal.dataset_source import DataFrameDatasetSource
from cleanlab_studio.internal.util import DatasetSourceType, init_dataset_source
from cleanlab_studio.internal.settings import CleanlabSettings
from cleanlab_studio.internal.types import FieldSchemaDict


class Studio:
    _api_key: str

    def __init__(self, api_key: Optional[str]):
        api.check_client_version()
        if api_key is None:
            try:
                api_key = CleanlabSettings.load().api_key
                if api_key is None:
                    raise ValueError
            except (FileNotFoundError, KeyError, ValueError):
                raise ValueError(
                    "No API key found; either specify API key or log in with 'cleanlab login' first"
                )
        api.validate_api_key(api_key)
        self._api_key = api_key

    def upload_dataset(
        self,
        dataset: DatasetSourceType,
        dataset_name: Optional[str] = None,
        *,
        schema_overrides: Optional[FieldSchemaDict] = None,
        modality: Optional[str] = None,
        id_column: Optional[str] = None,
    ) -> str:
        ds = init_dataset_source(dataset, dataset_name)
        if isinstance(ds, DataFrameDatasetSource) and dataset_name is None:
            raise ValueError("Must provide dataset name if uploading from a DataFrame")
        return upload.upload_dataset(
            self._api_key,
            ds,
            schema_overrides=schema_overrides,
            modality=modality,
            id_column=id_column,
        )
