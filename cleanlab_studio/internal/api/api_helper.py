import uuid
from typing import List, Optional, TypedDict

import requests

from cleanlab_studio.errors import (
    APIError,
    InvalidProjectConfiguration,
    InvalidUUIDError,
)
from cleanlab_studio.internal.types import JSONDict


class UploadPart(TypedDict):
    ETag: str
    PartNumber: int


UploadParts = List[UploadPart]


def check_uuid_well_formed(uuid_string: str, id_name: str) -> None:
    try:
        uuid.UUID(uuid_string)
    except ValueError:
        raise InvalidUUIDError(
            f"{uuid_string} is not a well-formed {id_name}, please double check and try again."
        )


def construct_headers(
    api_key: Optional[str], content_type: Optional[str] = "application/json"
) -> JSONDict:
    retval = dict()
    if api_key:
        retval["Authorization"] = f"bearer {api_key}"
    if content_type:
        retval["Content-Type"] = content_type
    retval["Client-Type"] = "python-api"
    return retval


def handle_api_error(res: requests.Response) -> None:
    handle_api_error_from_json(res.json(), res.status_code)


def handle_api_error_from_json(res_json: JSONDict, status_code: Optional[int] = None) -> None:
    if "code" in res_json and "description" in res_json:  # AuthError or UserQuotaError format
        if res_json["code"] == "user_soft_quota_exceeded":
            pass  # soft quota limit is going away soon, so ignore it
        else:
            raise APIError(res_json["description"])

    if res_json.get("error", None) is not None:
        error = res_json["error"]
        if (
            status_code == 422
            and isinstance(error, dict)
            and error.get("code", None) == "UNSUPPORTED_PROJECT_CONFIGURATION"
        ):
            raise InvalidProjectConfiguration(error["description"])
        raise APIError(res_json["error"])
