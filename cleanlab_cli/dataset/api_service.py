"""
Methods for interacting with the command line server API
1:1 mapping with command_line_api.py
"""
import requests

base_url = "localhost:8500/api/cli"


def initialize_dataset(
    api_key, fields, data_types, feature_types, id_column, modality, dataset_name
):
    return requests.post(
        base_url + "/initialize",
        data={
            "api_key": api_key,
            "fields": fields,
            "data_types": data_types,
            "feature_types": feature_types,
            "id_column": id_column,
            "modality": modality,
            "dataset_name": dataset_name,
        },
    )


def get_existing_ids(api_key, dataset_id):
    return requests.get(
        base_url + "/existing_ids", data={"api_key": api_key, "dataset_id": dataset_id}
    )


def get_dataset_schema(api_key, dataset_id):
    return requests.get(base_url + "/schema", data={"api_key": api_key, "dataset_id": dataset_id})


def upload_rows(api_key, dataset_id, rows):
    return requests.post(
        base_url + "/upload", data={"api_key": api_key, "dataset_id": dataset_id, "rows": rows}
    )


def complete_upload(api_key, dataset_id):
    return requests.put(base_url + "/complete", data={"api_key": api_key, "dataset_id": dataset_id})
