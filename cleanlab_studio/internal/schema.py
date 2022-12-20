from dataclasses import dataclass
from enum import Enum
import decimal
import os.path
import pathlib
import random
import re
from typing import Dict, Set, Optional, Union, Any, Collection, Sized, Tuple, List

import pandas as pd
import numpy as np
from pandas import NaT

from cleanlab_studio.internal.types import Modality
from cleanlab_studio.internal.dataset import Dataset, PandasDataset
from cleanlab_studio.version import SCHEMA_VERSION

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

    @classmethod
    def infer(
        cls,
        dataset,
        name: str,
        columns: Optional[Collection[str]] = None,
        id_column: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> "Schema":
        """
        Infer a Schema from a pyspark DataFrame.
        """
        # XXX inefficient, materializes entire dataset in RAM in a Pandas DataFrame
        return propose_schema(
            PandasDataset(dataset.toPandas()),
            name=name,
            columns=columns,
            id_column=id_column,
            modality=modality,
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


def propose_schema(
    dataset: Dataset,
    name: str,
    columns: Optional[Collection[str]] = None,
    id_column: Optional[str] = None,
    modality: Optional[str] = None,
    filepath_column: Optional[str] = None,
    sample_size: int = 10000,
    max_rows_checked: int = 200000,
) -> Schema:
    """
    Generates a schema for a dataset based on a sample of the dataset's rows.

    Dataset columns with no name will not be included in the schema.

    :param filepath:
    :param columns: columns to generate a schema for
    :param id_column: ID column name
    :param name: name of dataset
    :param filepath_column: filepath column name, i.e. name of column holding media filepaths (needed if modality is image)
    :param modality: data modality
    :param sample_size: default of 1000
    :param max_rows_checked: max rows to sample from
    :return:

    """
    # The arguments are intended to be required for the command-line interface, but are optional for Cleanlab Studio.
    # fill optional arguments if necessary
    if columns is None:
        columns = dataset.get_columns()

    if modality is None:
        if len(columns) > 5:
            modality = Modality.tabular.value
        else:
            modality = Modality.text.value

    # dataset = []
    rows = []
    for idx, row in enumerate(dataset.read_streaming_values()):
        if idx >= max_rows_checked:
            break
        if idx < sample_size:
            rows.append(row)
        else:
            random_idx = random.randint(0, idx)
            if random_idx < sample_size:
                rows[random_idx] = row
    df = pd.DataFrame(data=rows, columns=list(columns))

    schema_dict = dict()
    fields_dict = dict()
    for column_name in columns:
        if column_name == "":
            continue
        column_values = list(df[column_name][~df[column_name].isna()])
        column_values = [v for v in column_values if v != ""]

        if len(column_values) == 0:  # all values in column are empty, give default string[text]
            fields_dict[column_name] = dict(
                data_type=DataType.string.value, feature_type=FeatureType.text.value
            )
            continue

        col_data_type, col_feature_type = infer_types(column_values)
        fields_dict[column_name] = dict(
            data_type=col_data_type.value, feature_type=col_feature_type.value
        )

    schema_dict["fields"] = fields_dict

    if id_column is None:
        id_columns = [
            k
            for k, spec in schema_dict["fields"].items()
            if spec["feature_type"] == FeatureType.identifier.value
        ]
        if len(id_columns) == 0:
            id_columns = list(columns)
        id_column = _find_best_matching_column("id", id_columns)
    else:
        if id_column not in columns:
            raise ValueError(f"ID column '{id_column}' does not exist in the dataset.")

    assert id_column is not None

    metadata: Dict[str, Optional[str]] = dict(
        id_column=id_column, modality=modality, name=name, filepath_column=filepath_column
    )
    return Schema.create(metadata=metadata, fields=fields_dict, version=SCHEMA_VERSION)


def infer_types(values: Collection[Any]) -> Tuple[DataType, FeatureType]:
    """
    Infer the data type and feature type of a collection of a values using simple heuristics.

    :param values: a Collection of data values (that are not null and not empty string)
    """
    counts = {DataType.string: 0, DataType.integer: 0, DataType.float: 0, DataType.boolean: 0}
    ID_RATIO_THRESHOLD = 0.97  # lowerbound
    CATEGORICAL_RATIO_THRESHOLD = 0.20  # upperbound

    ratio_unique = len(set(values)) / len(values)
    for v in values:
        if v == "":
            continue
        if isinstance(v, str):
            counts[DataType.string] += 1
        elif isinstance(v, float):
            counts[DataType.float] += 1
        elif isinstance(v, bool):  # must come before int: isinstance(True, int) evaluates to True
            counts[DataType.boolean] += 1
        elif isinstance(v, int):
            counts[DataType.integer] += 1
        elif isinstance(v, decimal.Decimal):  # loading from JSONs can produce Decimal values
            counts[DataType.float] += 1
        else:
            raise ValueError(f"Value {v} has an unrecognized type: {type(v)}")

    ratios: Dict[DataType, float] = {k: v / len(values) for k, v in counts.items()}
    max_count_type = max(ratios, key=lambda k: ratios[k])

    # preliminary check: ints/floats may be loaded as strings
    if max_count_type == DataType.string:
        if string_values_are_integers(values):
            max_count_type = DataType.integer
        elif string_values_are_floats(values):
            max_count_type = DataType.float

    if max_count_type == DataType.string:
        if string_values_are_datetime(values):
            return DataType.string, FeatureType.datetime
        # is string type
        if ratio_unique >= ID_RATIO_THRESHOLD:
            # almost all unique values, i.e. either ID, text
            if multiple_separate_words_detected(values):
                return DataType.string, FeatureType.text
            else:
                if _values_are_filepaths(values):
                    return DataType.string, FeatureType.filepath
                return DataType.string, FeatureType.identifier
        elif ratio_unique <= CATEGORICAL_RATIO_THRESHOLD:
            return DataType.string, FeatureType.categorical
        else:
            return DataType.string, FeatureType.text

    elif max_count_type == DataType.integer:
        if ratio_unique >= ID_RATIO_THRESHOLD:
            return DataType.integer, FeatureType.identifier
        elif ratio_unique <= CATEGORICAL_RATIO_THRESHOLD:
            return DataType.integer, FeatureType.categorical
        else:
            return DataType.integer, FeatureType.numeric
    elif max_count_type == DataType.float:
        return DataType.float, FeatureType.numeric
    elif max_count_type == DataType.boolean:
        return DataType.boolean, FeatureType.boolean
    else:
        return DataType.string, FeatureType.text


def _find_best_matching_column(target_column: str, columns: List[str]) -> Optional[str]:
    """
    Find the column from `columns` that is the closest match to the `target_col`.
    If no columns are likely, pick the first column of `columns`
    If no columns are provided, return None

    :param target_column: some reserved column name, typically: 'id', 'label', or 'text'
    :param columns: non-empty list of column names
    :return:
    """
    if len(columns) == 0:
        return None

    if target_column == "id":
        regex = r"id"
    elif target_column == "filepath":
        regex = r"file|path|dir"
    else:
        regex = r""

    poss = []
    for c in columns:
        if c.lower() == target_column:
            return c
        elif any(re.findall(regex, c, re.IGNORECASE)):
            poss.append(c)

    if len(poss) > 0:  # pick first possibility
        return poss[0]
    else:
        return columns[0]


def multiple_separate_words_detected(values: Collection[Any]) -> bool:
    avg_num_words = sum([len(str(v).split()) for v in values]) / len(values)
    return avg_num_words >= 3


def is_filepath(string: str, check_existing: bool = False) -> bool:
    if check_existing:
        return os.path.exists(string)
    return pathlib.Path(string).suffix != "" and " " not in string


def get_validation_sample_size(values: Sized) -> int:
    return min(20, len(values))


def string_values_are_datetime(values: Collection[Any]) -> bool:
    try:
        # check for datetime first
        val_sample = random.sample(list(values), get_validation_sample_size(values))
        for s in val_sample:
            res = pd.to_datetime(s)
            if res is NaT:
                raise ValueError
    except Exception:
        return False
    return True


def string_values_are_integers(values: Collection[Any]) -> bool:
    try:
        val_sample = random.sample(list(values), get_validation_sample_size(values))
        for s in val_sample:
            if str(int(s)) != s:
                return False
    except Exception:
        return False
    return True


def string_values_are_floats(values: Collection[Any]) -> bool:
    try:
        val_sample = random.sample(list(values), get_validation_sample_size(values))
        for s in val_sample:
            float(s)
    except Exception:
        return False
    return True


def _values_are_filepaths(values: Collection[Any]) -> bool:
    val_sample = random.sample(list(values), get_validation_sample_size(values))
    for s in val_sample:
        if not is_filepath(s):
            return False
    return True
