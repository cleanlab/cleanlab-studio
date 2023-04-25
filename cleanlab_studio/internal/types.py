from enum import Enum
from typing import Any, Dict, Optional, TypedDict


JSONDict = Dict[str, Any]


class Modality(Enum):
    text = "text"
    tabular = "tabular"
    image = "image"


MODALITIES = [m.value for m in Modality]


class CleanlabSettingsDict(TypedDict):
    version: Optional[str]
    api_key: Optional[str]
