from typing import Dict, Any, Literal, Optional, Union, List
from typing import TypedDict

JSONDict = Dict[str, Any]
Modality = Literal["text", "tabular"]
DataType = Literal["string", "integer", "float", "boolean"]
FeatureType = Literal["identifier", "categorical", "numeric", "text", "boolean", "datetime"]
IDType = Union[str, int]
DatasetFileExtensionType = Literal[".csv", ".xls", ".xlsx", ".json"]
ALLOWED_EXTENSIONS: List[DatasetFileExtensionType] = [".csv", ".xls", ".xlsx", ".json"]


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
