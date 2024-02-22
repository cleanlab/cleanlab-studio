from typing import Any, Dict, Optional, TypedDict, Literal


JSONDict = Dict[str, Any]
FieldSchemaDict = Dict[str, Dict[str, Any]]
TLMQualityPreset = Literal["best", "high", "medium", "low", "base"]
TLMModel = Literal["gpt-3.5-turbo-16k", "gpt-4"]


class CleanlabSettingsDict(TypedDict):
    version: Optional[str]
    api_key: Optional[str]
