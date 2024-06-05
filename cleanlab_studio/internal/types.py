from typing import Any, Dict, Optional, TypedDict, Literal


JSONDict = Dict[str, Any]


class SchemaOverride(TypedDict):
    name: str
    column_type: str


TLMQualityPreset = Literal["best", "high", "medium", "low", "base"]


class CleanlabSettingsDict(TypedDict):
    version: Optional[str]
    api_key: Optional[str]
