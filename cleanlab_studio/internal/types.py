from typing import Any, Dict, Optional, TypedDict, Literal, Union
from cleanlab_studio.studio.trustworthy_language_model import TLMScore

JSONDict = Dict[str, Any]


class SchemaOverride(TypedDict):
    name: str
    column_type: str


TLMQualityPreset = Literal["best", "high", "medium", "low", "base"]

TLMScoreResponse = Union[float, TLMScore]


class CleanlabSettingsDict(TypedDict):
    version: Optional[str]
    api_key: Optional[str]
