from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import List

from cleanlab_studio.internal.api.beta_api import (
    complete_upload,
    get_dataset,
    initialize_upload,
    list_datasets,
)
from cleanlab_studio.internal.dataset_source import FilepathDatasetSource
from cleanlab_studio.internal.upload_helpers import upload_file_parts


@dataclass
class BetaDataset:
    id: str
    filename: str
    upload_complete: bool
    upload_date: int

    @classmethod
    def from_id(cls, api_key: str, dataset_id: str) -> "BetaDataset":
        dataset = get_dataset(api_key, dataset_id)
        return cls(
            id=dataset_id,
            filename=dataset["filename"],
            upload_complete=dataset["complete"],
            upload_date=dataset["upload_date"],
        )

    @classmethod
    def from_filepath(cls, api_key: str, filepath: str) -> "BetaDataset":
        dataset_source = FilepathDatasetSource(filepath=pathlib.Path(filepath))
        initialize_response = initialize_upload(
            api_key,
            dataset_source.get_filename(),
            dataset_source.file_type,
            dataset_source.file_size,
        )
        dataset_id = initialize_response["id"]
        part_sizes = initialize_response["part_sizes"]
        presigned_posts = initialize_response["presigned_posts"]

        # TODO: upload file parts
        upload_parts = upload_file_parts(dataset_source, part_sizes, presigned_posts)
        dataset = complete_upload(api_key, dataset_id, upload_parts)
        return cls(
            id=dataset_id,
            filename=dataset["filename"],
            upload_complete=dataset["complete"],
            upload_date=dataset["upload_date"],
        )

    @classmethod
    def list(cls, api_key: str) -> List[BetaDataset]:
        datasets = list_datasets(api_key)
        return [
            cls(
                id=dataset["id"],
                filename=dataset["filename"],
                upload_complete=dataset["complete"],
                upload_date=dataset["upload_date"],
            )
            for dataset in datasets
        ]
