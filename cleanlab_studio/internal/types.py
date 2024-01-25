from typing import Any, Dict, Optional, TypedDict


JSONDict = Dict[str, Any]


class SchemaOverride(TypedDict):
    name: str
    column_type: str


class CleanlabSettingsDict(TypedDict):
    version: Optional[str]
    api_key: Optional[str]
