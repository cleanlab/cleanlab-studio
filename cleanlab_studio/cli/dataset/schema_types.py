from dataclasses import dataclass
from enum import Enum
from typing import Dict, Set, Optional, Union, Any

from cleanlab_studio.cli.types import Modality

SchemaMetadataDictType = Dict[str, Optional[str]]
SchemaFieldsDictType = Dict[str, Dict[str, str]]


class DataType(Enum):
    string = "string"
    integer = "integer"
    float = "float"
    boolean = "boolean"


class FeatureType(Enum):
    identifier = "identifier"
    categorical = "categorical"
    numeric = "numeric"
    text = "text"
    boolean = "boolean"
    datetime = "datetime"
    filepath = "filepath"


DATA_TYPES_TO_FEATURE_TYPES: Dict[DataType, Set[FeatureType]] = {
    DataType.string: {
        FeatureType.text,
        FeatureType.categorical,
        FeatureType.datetime,
        FeatureType.identifier,
        FeatureType.filepath,
    },
    DataType.integer: {
        FeatureType.categorical,
        FeatureType.datetime,
        FeatureType.identifier,
        FeatureType.numeric,
    },
    DataType.float: {FeatureType.datetime, FeatureType.numeric},
    DataType.boolean: {FeatureType.boolean},
}


@dataclass
class FieldSpecification:
    data_type: DataType
    feature_type: FeatureType

    @staticmethod
    def create(data_type: str, feature_type: str) -> "FieldSpecification":
        data_type_ = DataType(data_type)
        feature_type_ = FeatureType(feature_type)

        if feature_type_ not in DATA_TYPES_TO_FEATURE_TYPES[data_type_]:
            raise ValueError(
                f"Invalid column feature type: '{feature_type_.value}' for data type: '{data_type_.value}'. "
                f"Accepted categories for type '{data_type_.value}' are: {', '.join(t.value for t in DATA_TYPES_TO_FEATURE_TYPES[data_type_])}"
            )
        return FieldSpecification(data_type=data_type_, feature_type=feature_type_)

    def to_dict(self) -> Dict[str, str]:
        return dict(data_type=self.data_type.value, feature_type=self.feature_type.value)


@dataclass
class SchemaMetadata:
    id_column: str
    modality: Modality
    name: str
    filepath_column: Optional[str]

    @staticmethod
    def create(
        id_column: str, modality: str, name: str, filepath_column: Optional[str] = None
    ) -> "SchemaMetadata":
        return SchemaMetadata(
            id_column=id_column,
            modality=Modality(modality),
            name=name,
            filepath_column=filepath_column,
        )

    def to_dict(self) -> SchemaMetadataDictType:
        return dict(
            id_column=self.id_column,
            modality=self.modality.value,
            name=self.name,
            filepath_column=self.filepath_column,
        )


@dataclass
class Schema:
    metadata: SchemaMetadata
    fields: Dict[str, FieldSpecification]
    version: str

    @staticmethod
    def create(
        metadata: SchemaMetadataDictType, fields: Dict[str, Dict[str, str]], version: str
    ) -> "Schema":
        fields_ = dict()
        for field, field_spec in fields.items():
            if not isinstance(field, str):
                raise ValueError(
                    f"All schema columns must be strings. Found non-string column: {field}"
                )
            if field == "":
                raise ValueError(
                    "Found empty string for schema column name. Schema columns cannot be empty strings."
                )
            fields_[field] = FieldSpecification.create(
                data_type=field_spec["data_type"], feature_type=field_spec["feature_type"]
            )

        # metadata variables
        id_column = metadata["id_column"]
        modality = metadata["modality"]
        name = metadata["name"]
        assert id_column is not None
        assert modality is not None
        assert name is not None
        filepath_column: Optional[str] = metadata.get("filepath_column", None)
        return Schema(
            metadata=SchemaMetadata.create(
                id_column=id_column, modality=modality, name=name, filepath_column=filepath_column
            ),
            fields=fields_,
            version=version,
        )

    def to_dict(self) -> Dict[str, Any]:
        retval: Dict[str, Any] = dict()
        retval["metadata"] = self.metadata.to_dict()
        fields: Dict[str, Dict[str, str]] = dict()
        for field, field_spec in self.fields.items():
            fields[field] = field_spec.to_dict()
        retval["fields"] = fields
        retval["version"] = self.version

        return retval

    @classmethod
    def from_dict(cls, schema_dict: Dict[str, Any]) -> "Schema":
        return Schema.create(
            metadata=schema_dict["metadata"],
            fields=schema_dict["fields"],
            version=schema_dict["version"],
        )


PYTHON_TYPES_TO_READABLE_STRING: Dict[type, str] = {
    str: DataType.string.value,
    float: DataType.float.value,
    int: DataType.integer.value,
    bool: DataType.boolean.value,
}

DATA_TYPES_TO_PYTHON_TYPES: Dict[DataType, type] = {
    DataType.string: str,
    DataType.float: float,
    DataType.integer: int,
    DataType.boolean: bool,
}
