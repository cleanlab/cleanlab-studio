"""
Helper functions for working with schemas
"""

import json
from decimal import Decimal
import random
from typing import (
    Any,
    Optional,
    Dict,
    List,
    Collection,
)

import pandas as pd
from pandas import NaT
import semver

from cleanlab_cli import MIN_SCHEMA_VERSION, SCHEMA_VERSION, MAX_SCHEMA_VERSION
from cleanlab_cli.click_helpers import progress, success, info, abort
from cleanlab_cli.dataset.schema_types import (
    DATA_TYPES_TO_FEATURE_TYPES,
)
from cleanlab_cli.util import (
    init_dataset_from_filepath,
    get_filename,
    dump_json,
)
import time

ALLOWED_EXTENSIONS = [".csv", ".xls", ".xlsx"]


def _find_best_matching_column(target_column: str, columns: List[str]) -> Optional[str]:
    """
    Find the column from `columns` that is the closest match to the `target_col`.
    If no columns are likely, pick the first column of `columns`

    :param target_column: some reserved column name, typically: 'id', 'label', or 'text'
    :param columns: non-empty list of column names
    :return:
    """
    assert len(columns) > 0, "list of columns is empty"
    poss = []
    for c in columns:
        if c.lower() == target_column:
            return c
        elif c.lower().endswith(f"_{target_column}"):
            poss.append(c)
        elif c.lower().startswith(f"{target_column}_"):
            poss.append(c)

    if len(poss) > 0:  # pick first possibility
        return poss[0]
    else:
        return columns[0]


def load_schema(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


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

    # check schema version validity
    schema_version = schema["version"]
    if semver.compare(MIN_SCHEMA_VERSION, schema_version) == 1:  # min schema > schema_version
        raise ValueError(
            "This schema version is incompatible with this version of the CLI. "
            "A new schema should be generated using 'cleanlab dataset schema generate'"
        )
    elif semver.compare(MAX_SCHEMA_VERSION, schema_version) == -1:
        raise ValueError(
            "CLI is not up to date with your schema version. Run 'pip install --upgrade cleanlab-cli'."
        )

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

    recognized_column_types = {"string", "integer", "float", "boolean", "datetime"}

    ## Check that each field has a feature_type that matches the base type
    for spec in schema["fields"].values():
        column_type = spec["data_type"]
        column_feature_type = spec.get("feature_type", None)
        if column_type not in recognized_column_types:
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
        raise ValueError(
            f"ID column field {id_column_name} must have feature type: 'identifier', but has"
            f" feature type: '{id_column_spec_feature_type}'"
        )

    ## Check that there exists at least one categorical column (to be used as label)
    has_categorical = any(
        spec["feature_type"] == "categorical" for spec in schema["fields"].values()
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
            int(spec["feature_type"] in variable_fields) for spec in schema["fields"].values()
        )
        if num_variable_columns < 2:
            raise ValueError(
                "Dataset modality is tabular; there must be at least one categorical field and one"
                " other variable field (i.e. categorical, numeric, or datetime)."
            )

    ## If text modality, check that at least one column has feature type 'text'
    elif modality == "text":
        has_text = any(spec["feature_type"] == "text" for spec in schema["fields"].values())
        if not has_text:
            raise ValueError("Dataset modality is text, but none of the fields is a text column.")
    else:
        raise ValueError(f"Unsupported dataset modality: {modality}")


def multiple_separate_words_detected(values):
    avg_num_words = sum([len(str(v).split()) for v in values]) / len(values)
    return avg_num_words >= 3


def _values_are_datetime(values):
    try:
        # check for datetime first
        val_sample = random.sample(list(values), 20)
        for s in val_sample:
            res = pd.to_datetime(s)
            if res is NaT:
                raise ValueError
    except Exception:
        return False
    return True


def _values_are_integers(values):
    try:
        val_sample = random.sample(list(values), 20)
        for s in val_sample:
            if str(int(s)) != s:
                return False
    except Exception:
        return False
    return True


def _values_are_floats(values):
    try:
        val_sample = random.sample(list(values), 20)
        for s in val_sample:
            float(s)
    except Exception:
        return False
    return True


def infer_types(values: Collection[Any]):
    """
    Infer the data type and feature type of a collection of a values using simple heuristics.

    :param values: a Collection of data values
    """
    counts = {"string": 0, "integer": 0, "float": 0, "boolean": 0}
    ID_RATIO_THRESHOLD = 0.97  # lowerbound
    CATEGORICAL_RATIO_THRESHOLD = 0.20  # upperbound

    ratio_unique = len(set(values)) / len(values)
    for v in values:
        if v == "":
            continue
        if isinstance(v, str):
            counts["string"] += 1
        elif isinstance(v, float) or isinstance(v, Decimal):
            counts["float"] += 1
        elif isinstance(v, int):
            counts["integer"] += 1
        elif isinstance(v, bool):
            counts["boolean"] += 1
        else:
            raise ValueError(f"Value {v} has an unrecognized type: {type(v)}")

    ratios: Dict[str, float] = {k: v / len(values) for k, v in counts.items()}
    max_count_type = max(ratios.items(), key=lambda kv: kv[1])[0]

    # preliminary check: ints/floats may be loaded as strings
    if max_count_type:
        if _values_are_integers(values):
            max_count_type = "integer"
        elif _values_are_floats(values):
            max_count_type = "float"

    if max_count_type == "string":
        if _values_are_datetime(values):
            return "string", "datetime"
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
    sample_size: int = 10000,
    max_rows_checked: int = 200000,
) -> Dict[str, str]:
    """
    Generates a schema for a dataset based on a sample of the dataset's rows.

    The arguments are intended to be required for the command-line interface, but optional for Cleanlab Studio.

    :param filepath:
    :param columns: columns to generate a schema for
    :param id_column: ID column name
    :param name: name of dataset
    :param modality: text or tabular
    :param sample_size: default of 1000
    :param max_rows_checked: max rows to sample from
    :return:

    """
    dataset = init_dataset_from_filepath(filepath)

    # fill optional arguments if necessary
    if columns is None:
        columns = dataset.get_columns()

    if name is None:
        name = get_filename(filepath)

    if modality is None:
        if len(columns) > 5:
            modality = "tabular"
        else:
            modality = "text"

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
    df = pd.DataFrame(data=rows, columns=columns)
    retval: Dict[str, Any] = dict()
    retval["metadata"] = {}
    retval["fields"] = {}

    for column_name in columns:
        column_values = list(df[column_name][~df[column_name].isna()])
        column_values = [v for v in column_values if v != ""]

        if len(column_values) == 0:  # all values in column are empty, give default string[text]
            retval["fields"][column_name] = {"data_type": "string", "feature_type": "text"}
            continue

        col_data_type, col_feature_type = infer_types(column_values)

        field_spec = {"data_type": col_data_type, "feature_type": col_feature_type}

        if col_feature_type is None:
            del field_spec["feature_type"]

        retval["fields"][column_name] = field_spec

    if id_column is None:
        id_columns = [
            k for k, spec in retval["fields"].items() if spec["feature_type"] == "identifier"
        ]
        if len(id_columns) == 0:
            id_columns = list(columns)
        id_column = _find_best_matching_column("id", id_columns)
    else:
        if id_column not in columns:
            abort(f"ID column '{id_column}' does not exist in the dataset.")

    retval["metadata"] = {"id_column": id_column, "modality": modality, "name": name}
    retval["version"] = SCHEMA_VERSION
    return retval


def construct_schema(fields, data_types, feature_types, id_column, modality, dataset_name):
    retval = {
        "fields": {},
        "metadata": {"id_column": id_column, "modality": modality, "name": dataset_name},
        "version": SCHEMA_VERSION,
    }
    for field, data_type, feature_type in zip(fields, data_types, feature_types):
        retval["fields"][field] = {"data_type": data_type, "feature_type": feature_type}
    return retval


def save_schema(schema, filename: Optional[str]):
    """

    :param schema:
    :param filename: filename to save schema with
    :return:
    """
    if filename == "":
        filename = "schema.json"
    if filename:
        progress(f"Writing schema to {filename}...")
        dump_json(filename, schema)
        success("Saved.")
    else:
        info("Schema was not saved.")
