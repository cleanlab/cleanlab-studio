"""
Contains utility functions for interacting with files and schemas
"""
import click
import pandas as pd
from pandas import NaT
from typing import (
    Tuple,
    Optional,
    Iterable,
    Dict,
    List,
    Collection,
    Set,
    Generator,
    Any,
)
from random import sample, random
import pyexcel
import os
from collections import defaultdict, OrderedDict
import pathlib
from sqlalchemy import Integer, String, Boolean, DateTime, Float, BigInteger
import json
from sys import getsizeof
from enum import Enum

ALLOWED_EXTENSIONS = [".csv", ".xls", ".xlsx"]
SCHEMA_VERSION = "1.0"  # TODO use package version no.

schema_mapper = {
    "string": String(),
    "integer": BigInteger(),
    "float": Float(),
    "boolean": Boolean(),
    "datetime": DateTime(),
}

DATA_TYPES_TO_FEATURE_TYPES = {
    "string": {"text", "categorical", "datetime", "identifier"},
    "integer": {"categorical", "datetime", "identifier", "numeric"},
    "float": {"datetime", "numeric"},
    "boolean": {"boolean"},
}

PYTHON_TYPES_TO_READABLE_STRING = {str: "string", float: "float", int: "integer", bool: "boolean"}


class ValidationWarning(Enum):
    MISSING_ID = 1
    MISSING_VAL = 2
    TYPE_MISMATCH = 3
    DUPLICATE_ID = 4


def get_value_type(val):
    for python_type, readable_string in PYTHON_TYPES_TO_READABLE_STRING.items():
        if isinstance(val, python_type):
            return readable_string
    return "unrecognized"


def get_file_extension(filename):
    file_extension = pathlib.Path(filename).suffix
    if file_extension in ALLOWED_EXTENSIONS:
        return file_extension
    raise ValueError(f"File extension for {filename} did not match allowed extensions.")


def is_allowed_extension(filename):
    return any([filename.endswith(ext) for ext in ALLOWED_EXTENSIONS])


def get_filename(filepath):
    return os.path.split(filepath)[-1]


def read_file_as_df(filepath):
    ext = get_file_extension(filepath)
    if ext == ".json":
        df = pd.read_json(filepath, convert_axes=False, convert_dates=False).T
        df.index = df.index.astype("str")
        df["id"] = df.index
    elif ext in ".csv":
        df = pd.read_csv(filepath, keep_default_na=True)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(filepath, keep_default_na=True)
    elif ext in [".parquet"]:
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Failed to read filetype: {ext}")
    return df


def is_null_value(val):
    return val is None or val == "" or pd.isna(val)


def get_num_rows(filepath: str):
    stream = read_file_as_stream(filepath)
    num_rows = 0
    for _ in stream:
        num_rows += 1
    return num_rows


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


def convert_schema_to_dtypes(schema):
    dtypes = {}
    # version = schema['version']
    for field in schema["fields"]:
        dtypes[field["name"]] = schema_mapper[field["value"].lower()]
    return dtypes


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
    :param columns:
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


def get_dataset_columns(filepath):
    stream = read_file_as_stream(filepath)
    for r in stream:
        return list(r.keys())


def read_file_as_stream(filepath) -> Generator[OrderedDict, None, None]:
    """
    Opens a file and reads it as a stream (aka row-by-row) to limit memory usage
    :param filepath: path to target file
    :return: a generator that yields dataset rows, with each row being an OrderedDict
    """
    ext = get_file_extension(filepath)

    if ext in [".csv", ".xls", ".xlsx"]:
        for r in pyexcel.iget_records(file_name=filepath):
            yield r


def validate_and_process_record(
    record,
    schema,
    seen_ids: Set[str],
    columns: Optional[List[str]] = None,
    existing_ids: Optional[Collection[str]] = None,
):
    """
    Validate the row against the provided schema; generate warnings where issues are found

    If row ID exists in `existing_ids`, the row has already been uploaded, so we return (None, row ID, None)

    If row ID exists in `seen_ids`, it is a duplicate row, so we return (None, row ID, warnings)

    If row ID is missing, we return (None, None, warnings)

    Otherwise, the only warnings will be for type mismatches and missing values, and we return
    (processed row, row ID, warnings), where warnings is an empty dict if no issues are found.

    :param record: a row in the dataset
    :param schema: dataset schema
    :param seen_ids: the set of row IDs that have been processed so far
    :param columns:
    :param existing_ids:
    :return: tuple (processed row: dict[str, any], row ID: optional[str], warnings: dict[str])
    """
    fields = schema["fields"]
    id_column = schema["metadata"]["id_column"]

    if columns is None:
        columns = list(fields)

    if existing_ids is None:
        existing_ids = set()

    row_id = record.get(id_column, None)

    if row_id == "" or row_id is None:
        return (
            None,
            None,
            {ValidationWarning.MISSING_ID.name: [f"Missing ID for record: {dict(record)}."]},
        )

    if row_id in existing_ids:
        return None, row_id, None

    if row_id in seen_ids:
        return (
            None,
            row_id,
            {
                ValidationWarning.DUPLICATE_ID.name: [
                    f"Duplicate ID found. ID '{row_id}' has already been encountered before."
                ]
            },
        )

    warnings = defaultdict(list)

    row = {c: record.get(c, None) for c in columns}
    for column_name, column_value in record.items():
        if column_name not in fields:
            continue
        col_type = fields[column_name]["data_type"]
        col_feature_type = fields[column_name]["feature_type"]

        warning = None
        if is_null_value(column_value):
            row[column_name] = None
            warning = f"{column_name}: value is missing", ValidationWarning.MISSING_VAL
        else:
            if col_feature_type == "datetime":
                try:
                    pd.to_datetime(column_value)
                except (ValueError, TypeError):
                    warning = (
                        f"{column_name}: expected datetime but unable to parse '{column_value}'"
                        f" with {get_value_type(column_value)} type. Datetime strings must be"
                        " parsable by pandas.to_datetime().",
                        ValidationWarning.TYPE_MISMATCH,
                    )
            else:
                if col_type == "string":
                    row[column_name] = str(column_value)  # type coercion
                elif col_type == "integer":
                    if not isinstance(column_value, int):
                        warning = (
                            f"{column_name}: expected 'int' but got '{column_value}' with"
                            f" {get_value_type(column_value)} type",
                            ValidationWarning.TYPE_MISMATCH,
                        )
                elif col_type == "float":
                    if not (isinstance(column_value, int) or isinstance(column_value, float)):
                        warning = (
                            f"{column_name}: expected 'int' but got '{column_value}' with"
                            f" {get_value_type(column_value)} type",
                            ValidationWarning.TYPE_MISMATCH,
                        )
                elif col_type == "boolean":
                    if not isinstance(column_value, bool):
                        col_val_lower = str(column_value).lower()
                        if col_val_lower in ["true", "t", "yes", "1"]:
                            row[column_name] = True
                        elif col_val_lower in ["false", "f", "no", "0"]:
                            row[column_name] = False
                        else:
                            warning = (
                                f"{column_name}: expected 'bool' but got '{column_value}' with"
                                f" {get_value_type(column_value)} type",
                                ValidationWarning.TYPE_MISMATCH,
                            )

        if warning:
            row[column_name] = None  # replace bad value with NULL
            msg, warn_type = warning
            warnings[warn_type.name].append(msg)

    return row, row_id, warnings


def upload_rows(
    filepath: str,
    schema: Dict[str, Any],
    existing_ids: Optional[Collection[str]] = None,
    payload_size: int = 10,
):
    """

    :param filepath: path to dataset file
    :param schema: a validated schema
    :param existing_ids:
    :param payload_size: size of each chunk of rows uploaded, in MB
    :return: None
    """
    columns = list(schema["fields"].keys())
    existing_ids = [] if existing_ids is None else set(existing_ids)

    rows = []
    row_size = None
    rows_per_payload = None
    seen_ids = set()

    for record in read_file_as_stream(filepath):
        row, row_id, warnings = validate_and_process_record(
            record, schema, seen_ids, columns, existing_ids
        )

        if row_id is None:
            click.secho(warnings[ValidationWarning.MISSING_ID.name][0])
            continue

        if row is None:  # could be duplicate or already uploaded
            if len(warnings) > 0:
                click.secho(warnings[ValidationWarning.DUPLICATE_ID.name][0])
            continue

        # row and row ID both present, i.e. row will be uploaded
        seen_ids.add(row_id)

        if row_size is None:
            row_size = getsizeof(row)
            rows_per_payload = int(payload_size * 10**6 / row_size)

        if warnings:
            missing_val_warnings = warnings.get(ValidationWarning.MISSING_VAL.name, [])
            for w in missing_val_warnings:
                click.secho(w)

            type_mismatch_warnings = warnings.get(ValidationWarning.TYPE_MISMATCH.name, [])
            for w in type_mismatch_warnings:
                click.secho(w)

        rows.append(row)
        if len(rows) >= rows_per_payload:
            # if sufficiently large, POST to API TODO
            click.secho("Uploading row chunk...", fg="blue")
            rows = []

    click.secho("Uploading final row chunk...", fg="blue")

    if len(rows) > 0:
        # TODO upload
        pass

    click.secho("Upload completed.", fg="green")
