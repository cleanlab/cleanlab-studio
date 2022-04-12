"""
Methods for interacting with the command line server API
1:1 mapping with command_line_api.py
"""
import requests
from cleanlab_cli.click_helpers import error, abort
import json

base_url = "http://localhost:8500/api/cli"


def handle_api_error(res: requests.Response):
    if res.json().get("error", None) is not None:
        abort(res.json()["error"])


def initialize_dataset(api_key, schema):
    fields = list(schema["fields"])
    data_types = [spec["data_type"] for spec in schema["fields"].values()]
    feature_types = [spec["feature_type"] for spec in schema["fields"].values()]
    id_column = schema["metadata"]["id_column"]
    modality = schema["metadata"]["modality"]
    dataset_name = schema["metadata"]["name"]

    res = requests.post(
        base_url + "/initialize",
        data={
            "api_key": api_key,
            "fields": json.dumps(fields),
            "data_types": json.dumps(data_types),
            "feature_types": json.dumps(feature_types),
            "id_column": id_column,
            "modality": modality,
            "dataset_name": dataset_name,
        },
    )
    handle_api_error(res)
    return res.json()["dataset_id"]


def get_existing_ids(api_key, dataset_id):
    res = requests.get(base_url + f"/existing_ids/{dataset_id}", data={"api_key": api_key})
    handle_api_error(res)
    return res.json()["existing_ids"]


def get_dataset_schema(api_key, dataset_id):
    res = requests.get(base_url + f"/schema/{dataset_id}", data={"api_key": api_key})
    handle_api_error(res)
    return res.json()["schema"]


def upload_rows(api_key, dataset_id, rows):
    res = requests.post(
        base_url + f"/upload/{dataset_id}",
        data={"api_key": api_key, "rows": json.dumps(rows)},
    )
    with open("temp.json", "w") as f:
        f.write(json.dumps(rows))
    handle_api_error(res)


def get_completion_status(api_key, dataset_id):
    res = requests.get(base_url + f"/complete/{dataset_id}", data={"api_key": api_key})
    handle_api_error(res)
    return res.json()["complete"]


def complete_upload(api_key, dataset_id):
    res = requests.patch(base_url + f"/complete/{dataset_id}", data={"api_key": api_key})
    handle_api_error(res)


def validate_api_key(api_key):
    res = requests.get(base_url + "/validate", data={"api_key": api_key})
    handle_api_error(res)
    return res.json()["valid"]
