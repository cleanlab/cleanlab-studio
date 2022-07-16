"""
Helper functions for processing and uploading dataset rows
"""
import asyncio
import threading
import queue
import aiohttp
import click
import json
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
from cleanlab_cli.util import (
    is_null_value,
    dump_json,
    init_dataset_from_filepath,
    get_file_size,
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
                    pd.Timestamp(column_value)
                except (ValueError, TypeError):
                    warning = (
                        f"{column_name}: expected datetime but unable to parse '{column_value}'"
                        f" with {get_value_type(column_value)} type. Datetime strings must be"
                        " parsable by pandas.Timestamp().",
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


def validate_rows(
    dataset_filepath: str,
    columns: List[str],
    schema: Dict[str, Any],
    log: dict,
    upload_queue: queue.Queue,
    existing_ids: Optional[Collection[str]] = None,
):
    """Iterates through dataset and validates rows. Places validated rows in upload queue.

    :param dataset_filepath: file path to load dataset from
    :param columns: list of column identifiers for dataset
    :param schema: a validated schema
    :param log: log dict to add warnings to
    :param upload_queue: queue to place validated rows in for upload
    :param existing_ids: set of row IDs that were already uploaded, defaults to None
    """
    existing_ids = set() if existing_ids is None else set([str(x) for x in existing_ids])
    seen_ids: Set[str] = set()

    dataset = init_dataset_from_filepath(dataset_filepath)
    num_records = len(dataset)

    for record in tqdm(
        dataset.read_streaming_records(), total=num_records, initial=1, leave=True, unit=" rows"
    ):
        row, row_id, warnings = validate_and_process_record(
            record, schema, seen_ids, existing_ids, columns
        )

        update_log_with_warnings(log, row_id, warnings)

        # row and row ID both present, i.e. row will be uploaded
        seen_ids.add(row_id)

        if row:
            upload_queue.put(list(row.values()), block=True)

    upload_queue.put(None, block=True)


async def upload_rows(
    api_key: str,
    dataset_id: Optional[str],
    columns: List[str],
    upload_queue: queue.Queue,
    rows_per_payload: int,
):
    """Gets rows from upload queue and uploads to API.

    :param api_key: 32-character alphanumeric string
    :param dataset_id: dataset ID
    :param columns: list of column identifiers for dataset
    :param upload_queue: queue to get validated rows from
    :param rows_per_payload: number of rows to upload per payload/chunk
    """
    columns_json: str = json.dumps(columns)

    async with aiohttp.ClientSession() as session:
        payload = []
        upload_tasks = []
        first_upload = True

        row = upload_queue.get()
        while row is not None:
            payload.append(row)

            if len(payload) >= rows_per_payload:
                upload_tasks.append(
                    asyncio.create_task(
                        api_service.upload_rows_async(
                            session=session,
                            api_key=api_key,
                            dataset_id=dataset_id,
                            rows=payload,
                            columns_json=columns_json,
                        )
                    )
                )
                payload = []

                # avoid race condition when creating table
                if first_upload:
                    await upload_tasks[0]
                    upload_tasks = []
                    first_upload = False

            row = upload_queue.get()
            # yield control
            await asyncio.sleep(0)

        # upload remaining rows
        if len(payload) > 0:
            upload_tasks.append(
                api_service.upload_rows_async(
                    session=session,
                    api_key=api_key,
                    dataset_id=dataset_id,
                    rows=payload,
                    columns_json=columns_json,
                )
            )

        await asyncio.gather(*upload_tasks)


def upload_dataset(
    api_key: str,
    dataset_id: Optional[str],
    filepath: str,
    schema: Dict[str, Any],
    existing_ids: Optional[Collection[str]] = None,
    output: Optional[str] = None,
    payload_size: float = 10,
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

    log = create_feedback_log()

    file_size = get_file_size(filepath)
    api_service.check_dataset_limit(file_size, api_key=api_key, show_warning=False)

    # NOTE: makes simplifying assumption that first row size is representative of all row sizes
    row_size = getsizeof(next(init_dataset_from_filepath(filepath).read_streaming_records()))
    rows_per_payload = int(payload_size * 1e6 / row_size)
    upload_queue: queue.Queue = queue.Queue(maxsize=2 * rows_per_payload)

    # create validation process
    validation_thread = threading.Thread(
        target=validate_rows,
        kwargs={
            "dataset_filepath": filepath,
            "columns": columns,
            "schema": schema,
            "log": log,
            "upload_queue": upload_queue,
            "existing_ids": existing_ids,
        },
    )

    # start and join processes
    validation_thread.start()
    asyncio.run(
        upload_rows(
            api_key=api_key,
            dataset_id=dataset_id,
            columns=columns,
            upload_queue=upload_queue,
            rows_per_payload=rows_per_payload,
        )
    )
    validation_thread.join()

    api_service.complete_upload(api_key=api_key, dataset_id=dataset_id)

    # Check against soft quota, warn if applicable
    api_service.check_dataset_limit(file_size=0, api_key=api_key, show_warning=True)

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
        "Upload completed. View your uploaded dataset at https://app.cleanlab.ai",
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
