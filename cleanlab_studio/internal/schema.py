from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Set

from cleanlab_studio.internal.types import Modality
from cleanlab_studio.version import MAX_SCHEMA_VERSION, MIN_SCHEMA_VERSION

import numpy as np
import semver


SchemaMetadataDictType = Dict[str, Optional[str]]
SchemaFieldsDictType = Dict[str, Dict[str, str]]


class DataType(Enum):
    string = "string"
    integer = "integer"
    float = "float"
    boolean = "boolean"

    def as_numpy_type(self) -> Any:
        return {
            DataType.string: str,
            DataType.integer: np.int64,  # XXX backend might use big integers
            DataType.float: np.float64,
            DataType.boolean: bool,
        }[self]


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

    def validate(self) -> None:
        """
        Checks that:
        (1) all schema column names are strings
        (2) all schema column types are recognized
        Note that schema initialization already checks that all keys are present and that fields are valid.
        :param schema:
        :return: raises a ValueError if any checks fail
        """

        # check schema version validity
        if (
            semver.VersionInfo.parse(MIN_SCHEMA_VERSION).compare(self.version) == 1
        ):  # min schema > schema_version
            raise ValueError(
                "This schema version is incompatible with this version of the CLI. "
                "A new schema should be generated using 'cleanlab dataset schema generate'"
            )
        elif semver.VersionInfo.parse(MAX_SCHEMA_VERSION).compare(self.version) == -1:
            raise ValueError(
                "CLI is not up to date with your schema version. Run 'pip install --upgrade cleanlab-studio'."
            )

        # Advanced validation checks: this should be aligned with ConfirmSchema's validate() function
        ## Check that specified ID column has the feature_type 'identifier'
        id_column_name = self.metadata.id_column
        id_column_spec_feature_type = self.fields[id_column_name].feature_type
        if id_column_spec_feature_type != FeatureType.identifier:
            raise ValueError(
                f"ID column field {id_column_name} must have feature type: 'identifier', but has"
                f" feature type: '{id_column_spec_feature_type}'"
            )

        ## Check that there exists at least one categorical column (to be used as label)
        has_categorical = any(
            spec.feature_type == FeatureType.categorical for spec in self.fields.values()
        )
        if not has_categorical:
            raise ValueError(
                "Dataset does not seem to contain a label column. (None of the fields is categorical.)"
            )

        ## If tabular modality, check that there are at least two variable (i.e. categorical, numeric, datetime) columns
        modality = self.metadata.modality
        variable_fields = {FeatureType.categorical, FeatureType.numeric, FeatureType.datetime}
        if modality == Modality.tabular:
            num_variable_columns = sum(
                int(spec.feature_type in variable_fields) for spec in self.fields.values()
            )
            if num_variable_columns < 2:
                raise ValueError(
                    "Dataset modality is tabular; there must be at least one categorical field and one"
                    " other variable field (i.e. categorical, numeric, or datetime)."
                )

        ## If text modality, check that at least one column has feature type 'text'
        elif modality == Modality.text:
            has_text = any(spec.feature_type == FeatureType.text for spec in self.fields.values())
            if not has_text:
                raise ValueError(
                    "Dataset modality is text, but none of the fields is a text column."
                )

        elif modality == Modality.image:
            image_columns = [
                col for col, spec in self.fields.items() if spec.feature_type == FeatureType.image
            ]
            if not image_columns:
                raise ValueError(
                    "Dataset modality is image, but none of the fields is an image column."
                )
            if len(image_columns) > 1:
                raise ValueError(
                    "More than one image column in a dataset is not currently supported."
                )
