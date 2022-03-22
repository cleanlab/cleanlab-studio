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
from sqlalchemy import Integer, String, Boolean, DateTime, Float
import json
from sys import getsizeof

ALLOWED_EXTENSIONS = [".csv", ".xls", ".xlsx"]
SCHEMA_VERSION = "1.0"  # TODO use package version no.

schema_mapper = {
    "string": String(),
    "integer": Integer(),
    "float": Float(),
    "boolean": Boolean(),
    "datetime": DateTime(),
}

TYPE_TO_CATEGORIES = {
    "string": {"text", "categorical", "datetime", "id"},
    "integer": {"categorical", "datetime", "id", "numeric"},
    "float": {"datetime", "numeric"},
    "boolean": {"boolean"},
}


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
    return val is None or val == ""


def get_num_rows(filepath: str):
    stream = read_file_as_stream(filepath)
    num_rows = 0
    for _ in stream:
        num_rows += 1
    return num_rows


def _find_best_matching_column(target_col: str, columns: List[str]) -> Optional[str]:
    """
    Find the column from `columns` that is the closest match to the `target_col`.
    If no columns are likely, pick the first column of `columns` #TODO janky

    :param target_col: some reserved column name, typically: 'id', 'label', or 'text'
    :param columns: list of column names
    :return:
    """
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

    # Check that schema is complete with fields, metadata, and version
    for key in ["fields", "metadata", "version"]:
        if key not in schema:
            raise KeyError(f"Schema is missing '{key}' key.")

    schema_columns = set(schema["fields"])
    columns = set(columns)

    # Check that schema fields are strings
    for col in schema_columns:
        if not isinstance(col, str):
            raise ValueError(
                f"All schema columns must be strings. Found invalid column name: {col}"
            )

    # Check that the dataset has all columns specified in the schema
    if not schema_columns.issubset(columns):
        raise ValueError(f"Dataset is missing schema columns: {schema_columns - columns}")

    # Check that each field has a category that matches the base type
    for spec in schema["fields"].values():
        column_type = spec["type"]
        column_category = spec.get("category", None)
        if column_type not in schema_mapper:
            raise ValueError(f"Unrecognized column data type: {column_type}")

        if column_category:
            if column_category not in TYPE_TO_CATEGORIES[column_type]:
                raise ValueError(
                    f"Invalid column category: '{column_category}'. Accepted categories for type"
                    f" '{column_type}' are: {TYPE_TO_CATEGORIES[column_type]}"
                )

    # Check that metadata is complete
    metadata = schema["metadata"]
    for key in ["id_column", "modality", "name"]:
        if key not in metadata:
            raise KeyError(f"Metadata is missing the '{key}' key.")

    # Check that specified ID column has the category 'id'
    id_col_name = metadata["id_column"]
    id_col_spec_category = schema["fields"][id_col_name]["category"]
    if id_col_spec_category != "id":
        raise ValueError(
            f"The specified ID column '{id_col_name}' must have category 'id' in the schema fields."
        )


def multiple_separate_words_detected(values):
    avg_num_words = sum([len(str(v).split()) for v in values]) / len(values)
    return avg_num_words >= 3


def infer_type_and_category(values: Collection[any]):
    """
    Infer the type and category of a collection of a values using simple heuristics.

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
                return "string", "id"
        elif ratio_unique <= CATEGORICAL_RATIO_THRESHOLD:
            return "string", "categorical"
        else:
            return "string", "text"

    elif max_count_type == "integer":
        if ratio_unique >= ID_RATIO_THRESHOLD:
            return "integer", "id"
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
    retval["fields"] = {}

    for col_name in columns:
        col_vals = list(df[col_name][~df[col_name].isna()])
        col_vals = [v for v in col_vals if v != ""]

        if len(col_vals) == 0:  # all values in column are empty, give default string[text]
            retval["fields"][col_name] = {"type": "string", "category": "text"}
            continue

        col_type, col_category = infer_type_and_category(col_vals)

        field_spec = {"type": col_type, "category": col_category}

        if col_category is None:
            del field_spec["category"]

        retval["fields"][col_name] = field_spec

    if id_column is None:
        id_cols = [k for k, spec in retval["fields"].items() if spec["category"] == "id"]
        if len(id_cols) == 0:
            id_cols = columns
        id_column = _find_best_matching_column("id", id_cols)

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
    columns: Optional[List[str]] = None,
    existing_ids: Optional[Collection[str]] = None,
):
    fields = schema["fields"]
    id_col = schema["metadata"]["id_column"]

    if columns is None:
        columns = list(fields)
    if existing_ids is None:
        existing_ids = set()

    row_id = record[id_col]

    if row_id in existing_ids:
        return  # TODO should duplicate IDs be silent? Can't distinguish between resumes and actual duplicates

    if row_id == "":
        raise ValueError("Missing ID column field")

    row = {c: record[c] for c in columns}
    for col_name, col_val in record.items():
        col_type = fields[col_name]["type"]
        col_category = fields[col_name]["category"]

        if is_null_value(col_val):
            row[col_name] = None
            continue

        if col_category == "datetime":
            try:
                pd.to_datetime(col_val)
            except (ValueError, TypeError):
                raise TypeError(
                    f"Unable to parse '{col_val}' with type '{type(col_val)}'. "
                    "Datetime strings must be parsable by pandas.to_datetime()."
                )
        else:
            if col_type == "string":
                row[col_name] = str(col_type)  # type coercion
            elif col_type == "integer":
                if not isinstance(col_val, int):
                    raise TypeError(
                        f"Expected 'int' but got '{col_val}' with type '{type(col_val)}'"
                    )
            elif col_type == "float":
                if not (isinstance(col_val, int) or isinstance(col_val, float)):
                    raise TypeError(
                        f"Expected 'int' but got '{col_val}' with type '{type(col_val)}'"
                    )
            elif col_type == "boolean":
                if not isinstance(col_val, bool):
                    if col_val.lower() in ["true", "t", "yes"]:
                        row[col_name] = True
                    elif col_val.lower() == ["false", "f", "no"]:
                        row[col_name] = False
                    else:
                        raise TypeError(
                            f"Expected 'bool' but got '{col_val}' with type '{type(col_val)}'"
                        )
    return row


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
    fields = schema["fields"]
    columns = list(schema["fields"].keys())

    metadata = schema["metadata"]
    id_col = metadata["id_column"]
    modality = metadata["modality"]
    name = metadata["name"]
    existing_ids = [] if existing_ids is None else set(existing_ids)

    rows = []
    row_size = None
    rows_per_payload = None

    for record in read_file_as_stream(filepath):
        try:
            row = validate_and_process_record(record, schema, columns, existing_ids)
            if row_size is None:
                row_size = getsizeof(row)
                rows_per_payload = int(payload_size * 10**6 / row_size)
            # check passed
            rows.append(row)
            if len(rows) >= rows_per_payload:
                # if sufficiently large, POST to API TODO
                click.secho("Uploading row chunk...", fg="blue")
                rows = []
        except (KeyError, TypeError) as e:
            row_id = record.get(id_col, None)
            if row_id:
                click.secho(f"Invalid row. ID: ({row_id}). {str(e)}")
            else:
                click.secho(f"Invalid row with no ID column: {record}")

    click.secho("Uploading last row chunk...", fg="blue")

    if len(rows) > 0:
        # TODO upload
        pass

    click.secho("Upload completed.", fg="green")
