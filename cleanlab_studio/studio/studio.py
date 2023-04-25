from typing import Any, Optional


from . import upload
from cleanlab_studio.internal.api import api
from cleanlab_studio.internal.util import init_dataset_source
from cleanlab_studio.internal.schema import Schema
from cleanlab_studio.internal.settings import CleanlabSettings


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
        dataset: Any,  # for now, a spark.DataFrame; in the future, we will support more
        dataset_name: Optional[str] = None,
        schema: Optional[Schema] = None,
    ) -> str:
        ds = init_dataset_source(dataset, dataset_name)
        return upload.upload_tabular_dataset(self._api_key, ds, schema)

    # def upload_image_dataset(
    #     self,
    #     dataset: Any,  # spark.DataFrame
    #     id_column: str,
    #     path_column: str,
    #     content_column: str,
    #     label_column: str,
    #     *,
    #     dataset_name: Optional[str] = None,
    #     id: Optional[str] = None,  # for resuming upload
    # ) -> str:
    #     pass
