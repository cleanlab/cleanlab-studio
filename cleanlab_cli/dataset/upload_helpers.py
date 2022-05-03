"""
Helper functions for processing and uploading dataset rows
"""
import click
import pandas as pd
from decimal import Decimal
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
from tqdm import tqdm
from cleanlab_cli import api_service
from cleanlab_cli.dataset.util import (
    is_null_value,
    read_file_as_stream,
    dump_json,
    count_records_in_dataset_file,
)
from cleanlab_cli.dataset.schema_types import PYTHON_TYPES_TO_READABLE_STRING
from cleanlab_cli import click_helpers
from cleanlab_cli.click_helpers import success, info, progress


class ValidationWarning(Enum):
    MISSING_ID = 1
    MISSING_VAL = 2
    TYPE_MISMATCH = 3
    DUPLICATE_ID = 4


def warning_to_readable_name(warning: str):
    return {
        "MISSING_ID": "Rows with missing IDs (rows are dropped)",
        "MISSING_VAL": "Rows with missing values (values replaced with null)",
        "TYPE_MISMATCH": (
            "Rows with values that do not match the schema (values replaced with null)"
        ),
        "DUPLICATE_ID": (
            "Rows with duplicate IDs (only the first row instance is kept, all later rows dropped)"
        ),
    }[warning]


def get_value_type(val):
    for python_type, readable_string in PYTHON_TYPES_TO_READABLE_STRING.items():
        if isinstance(val, python_type):
            return readable_string
    return "unrecognized"


def validate_and_process_record(
    record,
    schema,
    seen_ids: Set[str],
    existing_ids: Set[str],
    columns: Optional[List[str]] = None,
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
    :return: tuple (processed row: dict[str, any], row ID: optional[str], warnings: dict[warn_type: str, desc: str])
    """
    fields = schema["fields"]
    id_column = schema["metadata"]["id_column"]

    if columns is None:
        columns = list(fields)

    row_id = record.get(id_column, None)

    if row_id == "" or row_id is None:
        return (
            None,
            None,
            {ValidationWarning.MISSING_ID.name: [f"Missing ID for record: {dict(record)}."]},
        )

    # row_id = str(row_id)
    if str(row_id) in existing_ids:
        return None, row_id, None

    if row_id in seen_ids:
        return (
            None,
            row_id,
            {ValidationWarning.DUPLICATE_ID.name: [f"Duplicate ID found: {dict(record)}"]},
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
                    if isinstance(column_value, Decimal):
                        row[column_name] = float(column_value)
                    else:
                        if not (isinstance(column_value, int) or isinstance(column_value, float)):
                            warning = (
                                f"{column_name}: expected 'float' but got '{column_value}' with"
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


def create_feedback_log():
    log = dict()
    log[ValidationWarning.MISSING_ID.name] = []
    # map from row ID to warnings
    log[ValidationWarning.DUPLICATE_ID.name] = dict()
    log[ValidationWarning.TYPE_MISMATCH.name] = dict()
    log[ValidationWarning.MISSING_VAL.name] = dict()
    return log


def update_log_with_warnings(log, row_id, warnings):
    if warnings:
        for warn_type in warnings:
            if warn_type == ValidationWarning.MISSING_ID.name:
                log[warn_type] += warnings[warn_type]
            else:
                log[warn_type][row_id] = warnings[warn_type]
    return log


def echo_log_warnings(log):
    for w in ValidationWarning:
        warning_count = len(log[w.name])
        if warning_count > 0:
            click.echo(f"{warning_to_readable_name(w.name)}: {warning_count}")


def upload_rows(
    api_key: str,
    dataset_id: Optional[str],
    filepath: str,
    schema: Dict[str, Any],
    existing_ids: Optional[Collection[str]] = None,
    output: Optional[str] = None,
    payload_size: float = 2,
):
    """

    :param api_key: 32-character alphanumeric string
    :param dataset_id: dataset ID
    :param filepath: path to dataset file
    :param schema: a validated schema
    :param existing_ids: set of row IDs that were already uploaded
    :param output: filepath to store upload issues in
    :param payload_size: size of each chunk of rows uploaded, in MB
    :return: None
    """
    columns = list(schema["fields"].keys())
    existing_ids = set() if existing_ids is None else set([str(x) for x in existing_ids])
    rows = []
    rows_per_payload = None
    seen_ids = set()

    log = create_feedback_log()

    num_records = count_records_in_dataset_file(filepath)
    for record in tqdm(
        read_file_as_stream(filepath), total=num_records, initial=1, leave=True, unit=" rows"
    ):
        row, row_id, warnings = validate_and_process_record(
            record, schema, seen_ids, existing_ids, columns
        )

        update_log_with_warnings(log, row_id, warnings)

        # row and row ID both present, i.e. row will be uploaded
        seen_ids.add(row_id)

        if row:
            # compute rows_per_payload if not available
            if rows_per_payload is None:
                row_size = getsizeof(row)
                rows_per_payload = int(payload_size * 10**6 / row_size)

            rows.append(row)
            if len(rows) >= rows_per_payload:
                # if sufficiently large, POST to API
                api_service.upload_rows(api_key=api_key, dataset_id=dataset_id, rows=rows)
                # click.secho("Uploading row chunk...", fg="blue")
                rows = []

    # click.secho("Uploading final row chunk...", fg="blue")
    if len(rows) > 0:
        api_service.upload_rows(api_key=api_key, dataset_id=dataset_id, rows=rows)

    api_service.complete_upload(api_key=api_key, dataset_id=dataset_id)

    total_warnings = sum([len(log[w.name]) for w in ValidationWarning])
    issues_found = total_warnings > 0
    if not issues_found:
        success("\nNo issues were encountered when uploading your dataset. Nice!")
    else:
        info(f"\n{total_warnings} issues were encountered when uploading your dataset.")
        echo_log_warnings(log)

        if not output:
            output = click_helpers.confirm_save_prompt_filepath(
                save_message="Would you like to save the issues for viewing?",
                save_default=None,
                prompt_message=(
                    "Specify a filename for the dataset issues. Leave this blank to use default"
                ),
                prompt_default="issues.json",
                no_save_message="Dataset type issues were not saved.",
            )
        # if we have an output after the above prompt (or originally provided)
        if output:
            save_feedback(log, output)
            click_helpers.confirm_open_file(
                "Would you like to open your issues file for viewing?", filepath=output
            )
    click.secho(
        "Upload completed. View your uploaded dataset at https://app.cleanlab.ai/datasets",
        fg="green",
    )


def group_feature_types(schema):
    """
    Given a schema, return a dict mapping each feature type to the list of columns with said feature type
    """
    feature_types_to_columns = defaultdict(list)
    for field_name, spec in schema["fields"].items():
        feature_type = spec["feature_type"]
        feature_types_to_columns[feature_type].append(field_name)
    return feature_types_to_columns


def save_feedback(feedback, filename):
    if not filename:
        raise ValueError("No filepath provided for saving feedback")
    feedback = {warning_to_readable_name(k): v for k, v in feedback.items()}
    progress(f"Writing issues to {filename}...")
    dump_json(filename, feedback)
    success("Saved.\n")
