from typing import Any, Dict, Optional, TypedDict, Literal


JSONDict = Dict[str, Any]
FieldSchemaDict = Dict[str, Dict[str, Any]]
TLMQualityPreset = Literal["best", "high", "medium", "low", "base"]


class CleanlabSettingsDict(TypedDict):
    version: Optional[str]
    api_key: Optional[str]
