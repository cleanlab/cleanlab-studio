from typing import Any, Dict, Optional, TypedDict


JSONDict = Dict[str, Any]
FieldSchemaDict = Dict[str, Dict[str, Any]]


class CleanlabSettingsDict(TypedDict):
    version: Optional[str]
    api_key: Optional[str]
