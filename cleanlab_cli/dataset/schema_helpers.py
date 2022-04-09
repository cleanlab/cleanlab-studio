"""
Helper functions for working with schemas
"""

import pandas as pd
from pandas import NaT
from typing import (
    Optional,
    Dict,
    List,
    Collection,
)
from random import sample, random
import json
from cleanlab_cli.dataset.util import (
    get_num_rows,
    get_dataset_columns,
    read_file_as_stream,
    get_filename,
)
from cleanlab_cli.dataset.schema_types import (
    schema_mapper,
    DATA_TYPES_TO_FEATURE_TYPES,
    SCHEMA_VERSION,
)

ALLOWED_EXTENSIONS = [".csv", ".xls", ".xlsx"]


def _find_best_matching_column(target_col: str, columns: List[str]) -> Optional[str]:
    """
    Find the column from `columns` that is the closest match to the `target_col`.
    If no columns are likely, pick the first column of `columns`

    :param target_col: some reserved column name, typically: 'id', 'label', or 'text'
    :param columns: non-empty list of column names
    :return:
    """
    assert len(columns) > 0, "list of columns is empty"
    poss = []
    for c in columns:
        if c.lower() == target_col:
            return c
        elif c.lower().endswith(f"_{target_col}"):
            poss.append(c)
        elif c.lower().startswith(f"{target_col}_"):
            poss.append(c)

    if len(poss) > 0:  # pick first possibility
        return poss[0]
    else:
        return columns[0]


def load_schema(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def dump_schema(filepath, schema):
    with open(filepath, "w") as f:
        f.write(json.dumps(schema, indent=2))


def validate_schema(schema, columns: Collection[str]):
    """
    Checks that:
    (1) all schema column names are strings
    (2) all schema columns exist in the dataset columns
    (3) all schema column types are recognized

    :param schema:
    :param columns: full list of columns in dataset
    :return: raises a ValueError if any checks fail
    """

    # Check for completeness and basic type correctness
    ## Check that schema is complete with fields, metadata, and version
    for key in ["fields", "metadata", "version"]:
        if key not in schema:
            raise KeyError(f"Schema is missing '{key}' key.")

    schema_columns = set(schema["fields"])
    columns = set(columns)

    ## Check that metadata is complete
    metadata = schema["metadata"]
    for key in ["id_column", "modality", "name"]:
        if key not in metadata:
            raise KeyError(f"Metadata is missing the '{key}' key.")

    ## Check that schema fields are strings
    for col in schema_columns:
        if not isinstance(col, str):
            raise ValueError(
                f"All schema columns must be strings. Found invalid column name: {col}"
            )

    ## Check that the dataset has all columns specified in the schema
    if not schema_columns.issubset(columns):
        raise ValueError(f"Dataset is missing schema columns: {schema_columns - columns}")

    ## Check that each field has a feature_type that matches the base type
    for spec in schema["fields"].values():
        column_type = spec["data_type"]
        column_feature_type = spec.get("feature_type", None)
        if column_type not in schema_mapper:
            raise ValueError(f"Unrecognized column data type: {column_type}")

        if column_feature_type:
            if column_feature_type not in DATA_TYPES_TO_FEATURE_TYPES[column_type]:
                raise ValueError(
                    f"Invalid column feature type: '{column_feature_type}'. Accepted categories for"
                    f" type '{column_type}' are: {DATA_TYPES_TO_FEATURE_TYPES[column_type]}"
                )

    # Advanced validation checks: this should be aligned with ConfirmSchema's validate() function
    ## Check that specified ID column has the feature_type 'identifier'
    id_column_name = metadata["id_column"]
    id_column_spec_feature_type = schema["fields"][id_column_name]["feature_type"]
    if id_column_spec_feature_type != "identifier":
        raise ValueError(f"ID column field {id_column_name} must have feature type: 'identifier'.")

    ## Check that there exists at least one categorical column (to be used as label)
    has_categorical = any(
        feature_type == "categorical" for feature_type in schema["fields"].values()
    )
    if not has_categorical:
        raise ValueError(
            "Dataset does not seem to contain a label column. (None of the fields is categorical.)"
        )

    ## If tabular modality, check that there are at least two variable (i.e. categorical, numeric, datetime) columns
    modality = metadata["modality"]
    variable_fields = {"categorical", "numeric", "datetime"}
    if modality == "tabular":
        num_variable_columns = sum(
            int(feature_type in variable_fields) for feature_type in schema["fields"].values()
        )
        if num_variable_columns < 2:
            raise ValueError(
                "Dataset modality is tabular; there must be at least one categorical field and one"
                " other variable field (i.e. categorical, numeric, or datetime)."
            )

    ## If text modality, check that at least one column has feature type 'text'
    if modality == "text":
        has_text = any(feature_type == "text" for feature_type in schema["fields"].values())
        if not has_text:
            raise ValueError("Dataset modality is text, but none of the fields is a text column.")


def multiple_separate_words_detected(values):
    avg_num_words = sum([len(str(v).split()) for v in values]) / len(values)
    return avg_num_words >= 3


def infer_types(values: Collection[any]):
    """
    Infer the data type and feature type of a collection of a values using simple heuristics.

    :param values: a Collection of data values
    """
    counts = {"string": 0, "integer": 0, "float": 0, "boolean": 0}
    ID_RATIO_THRESHOLD = 0.97  # lowerbound
    CATEGORICAL_RATIO_THRESHOLD = 0.20  # upperbound

    ratio_unique = len(set(values)) / len(values)
    for v in values:
        if isinstance(v, str):
            counts["string"] += 1
        elif isinstance(v, float):
            counts["float"] += 1
        elif isinstance(v, int):
            counts["integer"] += 1
        elif isinstance(v, bool):
            counts["boolean"] += 1
        else:
            raise ValueError(f"Value {v} has an unrecognized type: {type(v)}")

    ratios = {k: v / len(values) for k, v in counts.items()}
    types = list(ratios.keys())
    counts = list(ratios.values())
    max_count_type = types[counts.index(max(counts))]

    if max_count_type == "string":
        try:
            # check for datetime first
            val_sample = sample(list(values), 10)
            for s in val_sample:
                res = pd.to_datetime(s)
                if res is NaT:
                    raise ValueError
            return "string", "datetime"
        except (ValueError, TypeError):
            pass
        # is string type
        if ratio_unique >= ID_RATIO_THRESHOLD:
            # almost all unique values, i.e. either ID, text
            if multiple_separate_words_detected(values):
                return "string", "text"
            else:
                return "string", "identifier"
        elif ratio_unique <= CATEGORICAL_RATIO_THRESHOLD:
            return "string", "categorical"
        else:
            return "string", "text"

    elif max_count_type == "integer":
        if ratio_unique >= ID_RATIO_THRESHOLD:
            return "string", "identifier"  # identifiers are always strings
        elif ratio_unique <= CATEGORICAL_RATIO_THRESHOLD:
            return "integer", "categorical"
        else:
            return "integer", "numeric"
    elif max_count_type == "float":
        return "float", "numeric"
    elif max_count_type == "boolean":
        return "string", "categorical"
    else:
        return "string", "text"


def propose_schema(
    filepath: str,
    columns: Optional[Collection[str]] = None,
    id_column: Optional[str] = None,
    modality: Optional[str] = None,
    name: Optional[str] = None,
    num_rows: Optional[int] = None,
    sample_size: int = 1000,
) -> Dict[str, str]:
    """
    Generates a schema for a dataset based on a sample of up to 1000 of the dataset's rows.

    The arguments are intended to be required for the command-line interface, but optional for Cleanlab Studio.

    :param filepath:
    :param columns: columns to generate a schema for
    :param id_column: ID column name
    :param name: name of dataset
    :param modality: text or tabular
    :param num_rows: number of rows in dataset
    :param sample_size: default of 1000
    :return:

    """
    stream = read_file_as_stream(filepath)

    # fill optional arguments if necessary
    if columns is None:
        columns = get_dataset_columns(filepath)

    if num_rows is None:
        num_rows = get_num_rows(filepath)

    if name is None:
        name = get_filename(filepath)

    if modality is None:
        if len(columns) > 5:
            modality = "tabular"
        else:
            modality = "text"

    dataset = []
    sample_proba = 1 if sample_size >= num_rows else sample_size / num_rows
    for row in stream:
        if random() <= sample_proba:
            dataset.append(dict(row.items()))
    df = pd.DataFrame(dataset, columns=columns)
    retval = dict()
    retval["metadata"] = {}
    retval["fields"] = {}

    for col_name in columns:
        col_vals = list(df[col_name][~df[col_name].isna()])
        col_vals = [v for v in col_vals if v != ""]

        if len(col_vals) == 0:  # all values in column are empty, give default string[text]
            retval["fields"][col_name] = {"data_type": "string", "feature_type": "text"}
            continue

        col_data_type, col_feature_type = infer_types(col_vals)

        field_spec = {"data_type": col_data_type, "feature_type": col_feature_type}

        if col_feature_type is None:
            del field_spec["feature_type"]

        retval["fields"][col_name] = field_spec

    if id_column is None:
        id_columns = [
            k for k, spec in retval["fields"].items() if spec["feature_type"] == "identifier"
        ]
        if len(id_columns) == 0:
            id_columns = columns
        id_column = _find_best_matching_column("identifier", id_columns)

    retval["metadata"] = {"id_column": id_column, "modality": modality, "name": name}
    retval["version"] = SCHEMA_VERSION
    return retval


def construct_schema(fields, data_types, feature_types, id_column, modality, dataset_name):
    retval = {
        "fields": {},
        "metadata": {"id_column": id_column, "modality": modality, "name": dataset_name},
        "version": "1.0",  # TODO add package version
    }
    for field, data_type, feature_type in zip(fields, data_types, feature_types):
        retval["fields"][field] = {"data_type": data_type, "feature_type": feature_type}
    return retval
