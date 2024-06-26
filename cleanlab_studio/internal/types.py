from typing import Any, Dict, Optional, TypedDict, Literal, Union


JSONDict = Dict[str, Any]


class SchemaOverride(TypedDict):
    name: str
    column_type: str


TLMQualityPreset = Literal["best", "high", "medium", "low", "base"]

TLMScoreResponse = Union[float, Dict[str, Any]]


class CleanlabSettingsDict(TypedDict):
    version: Optional[str]
    api_key: Optional[str]
