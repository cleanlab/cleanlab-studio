"""
Helper functions for processing and uploading dataset rows
"""
import click
import pandas as pd
from typing import (
    Optional,
    Dict,
    List,
    Collection,
    Set,
    Any,
)
from collections import defaultdict
from sys import getsizeof
from enum import Enum
from cleanlab_cli import api_service
from cleanlab_cli.dataset.util import is_null_value, read_file_as_stream
from schema_types import PYTHON_TYPES_TO_READABLE_STRING, schema_mapper


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
    api_key: str,
    dataset_id: Optional[str],
    filepath: str,
    schema: Dict[str, Any],
    existing_ids: Optional[Collection[str]] = None,
    payload_size: int = 10,
):
    """

    :param api_key: 32-character alphanumeric string
    :param dataset_id: dataset ID
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
            # if sufficiently large, POST to API
            api_service.upload_rows(api_key=api_key, dataset_id=dataset_id, rows=rows)
            click.secho("Uploading row chunk...", fg="blue")
            rows = []

    click.secho("Uploading final row chunk...", fg="blue")
    if len(rows) > 0:
        api_service.upload_rows(api_key=api_key, dataset_id=dataset_id, rows=rows)

    api_service.complete_upload(api_key=api_key, dataset_id=dataset_id)
    click.secho("Upload completed.", fg="green")


def construct_sql_dtypes_from_schema(schema):
    return {field: schema_mapper[spec["data_type"]] for field, spec in schema["fields"].items()}


def group_feature_types(schema):
    """
    Given a schema, return a dict mapping each feature type to the list of columns with said feature type
    """
    feature_types_to_columns = defaultdict(list)
    for field_name, spec in schema["fields"].items():
        feature_type = spec["feature_type"]
        feature_types_to_columns[feature_type].append(field_name)
    return feature_types_to_columns
