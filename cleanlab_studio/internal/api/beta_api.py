from typing import Any, Dict, List

import requests

from .api import API_BASE_URL, construct_headers
from .api_helper import JSONDict, UploadParts, handle_api_error

experimental_jobs_base_url = f"{API_BASE_URL}/v0/experimental_jobs"


def initialize_upload(
    api_key: str, filename: str, file_type: str, file_size: int
) -> Dict[str, Any]:
    url = f"{experimental_jobs_base_url}/upload/initialize"
    headers = construct_headers(api_key)
    data = {
        "filename": filename,
        "file_type": file_type,
        "size_in_bytes": file_size,
    }
    resp = requests.post(url, headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()


def complete_upload(api_key: str, dataset_id: str, upload_parts: UploadParts) -> JSONDict:
    url = f"{experimental_jobs_base_url}/upload/complete"
    headers = construct_headers(api_key)
    data = {
        "dataset_id": dataset_id,
        "upload_parts": upload_parts,
    }
    resp = requests.post(url, headers=headers, json=data)
    handle_api_error(resp)
    return resp.json()


def get_dataset(api_key: str, dataset_id: str) -> JSONDict:
    url = f"{experimental_jobs_base_url}/datasets/{dataset_id}"
    headers = construct_headers(api_key)
    resp = requests.get(url, headers=headers)
    handle_api_error(resp)
    return resp.json()


def run_job(api_key: str, dataset_id: str, job_definition_name: str) -> JSONDict:
    url = f"{experimental_jobs_base_url}/run"
    headers = construct_headers(api_key)
    data = {
        "dataset_id": dataset_id,
        "job_definition_name": job_definition_name,
    }
    resp = requests.post(url, headers=headers, json=data)
    handle_api_error(resp)
    return resp.json()


def get_job(api_key: str, job_id: str) -> JSONDict:
    url = f"{experimental_jobs_base_url}/{job_id}"
    headers = construct_headers(api_key)
    resp = requests.get(url, headers=headers)
    handle_api_error(resp)
    return resp.json()


def get_job_status(api_key: str, job_id: str) -> JSONDict:
    url = f"{experimental_jobs_base_url}/{job_id}/status"
    headers = construct_headers(api_key)
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


def get_results(api_key: str, job_id: str) -> JSONDict:
    url = f"{experimental_jobs_base_url}/{job_id}/results"
    headers = construct_headers(api_key)
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


def list_datasets(api_key: str) -> List[JSONDict]:
    url = f"{experimental_jobs_base_url}/datasets"
    headers = construct_headers(api_key)
    resp = requests.get(url, headers=headers)
    handle_api_error(resp)
    return resp.json()["datasets"]


def list_jobs(api_key: str) -> List[JSONDict]:
    url = f"{experimental_jobs_base_url}/jobs"
    headers = construct_headers(api_key)
    resp = requests.get(url, headers=headers)
    handle_api_error(resp)
    return resp.json()["jobs"]
