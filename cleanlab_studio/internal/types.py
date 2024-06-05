from typing import Any, Dict, Literal, Optional, TypedDict, Union
from datetime import datetime

JSONDict = Dict[str, Any]


class SchemaOverride(TypedDict):
    name: str
    column_type: str


TLMQualityPreset = Literal["best", "high", "medium", "low", "base"]


class CleanlabSettingsDict(TypedDict):
    version: Optional[str]
    api_key: Optional[str]


class EnrichmentProjectFromDict(TypedDict):
    api_key: str
    id: str
    name: str
    target_column_in_dataset: str
    created_at: Optional[Union[str, datetime]]
