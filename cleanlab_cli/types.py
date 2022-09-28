from enum import Enum
from typing import Dict, Any, Optional, Union, TypedDict

JSONDict = Dict[str, Any]
IDType = Union[str, int]


class Modality(Enum):
    text = "text"
    tabular = "tabular"
    image = "image"


MODALITIES = [m.value for m in Modality]


class DatasetFileExtension(Enum):
    csv = ".csv"
    xls = ".xls"
    xlsx = ".xlsx"
    json = ".json"


class ImageFileExtension(Enum):
    jpeg = ".jpeg"
    png = ".png"


RecordType = Dict[str, Any]


class CommandState(TypedDict):
    command: Optional[str]
    args: Dict[str, Optional[str]]


class CleanlabSettingsDict(TypedDict):
    version: Optional[str]
    api_key: Optional[str]
