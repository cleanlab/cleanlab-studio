from enum import Enum
from typing import Dict, Any, Literal, Optional, Union, List, TypedDict

JSONDict = Dict[str, Any]
Modality = Literal["text", "tabular"]
DataType = Literal["string", "integer", "float", "boolean"]
FeatureType = Literal["identifier", "categorical", "numeric", "text", "boolean", "datetime"]
IDType = Union[str, int]
ValidationWarningType = Literal["MISSING_ID", "MISSING_VAL", "TYPE_MISMATCH", "DUPLICATE_ID"]

VALIDATION_WARNING_TYPES: List[ValidationWarningType] = [
    "MISSING_ID",
    "MISSING_VAL",
    "TYPE_MISMATCH",
    "DUPLICATE_ID",
]


class DatasetFileExtension(Enum):
    csv = ".csv"
    xls = ".xls"
    xlsx = ".xlsx"
    json = ".json"


RecordType = Dict[str, Any]

RowWarningsType = Dict[ValidationWarningType, List[str]]


class WarningLogType(TypedDict):
    MISSING_ID: List[str]
    MISSING_VAL: Dict[str, List[str]]
    TYPE_MISMATCH: Dict[str, List[str]]
    DUPLICATE_ID: Dict[str, List[str]]


class FieldSpecification(TypedDict):
    data_type: DataType
    feature_type: FeatureType


class SchemaMetadata(TypedDict):
    id_column: str
    modality: Modality
    name: str


class Schema(TypedDict):
    metadata: SchemaMetadata
    fields: Dict[str, FieldSpecification]
    version: str


class CommandState(TypedDict):
    command: Optional[str]
    args: Dict[str, Optional[str]]


class CleanlabSettingsDict(TypedDict):
    version: Optional[str]
    api_key: Optional[str]
