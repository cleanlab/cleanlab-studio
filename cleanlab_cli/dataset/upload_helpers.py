"""
Helper functions for processing and uploading dataset rows
"""
import asyncio
import threading
import queue
from asyncio import Task

import aiohttp
import click
import json
import pandas as pd
import re
from decimal import Decimal
from typing import (
    Optional,
    Dict,
    List,
    Collection,
    Set,
    Any,
    Tuple,
    Coroutine,
    Union,
)
from collections import defaultdict
from sys import getsizeof
from enum import Enum
from tqdm import tqdm
from cleanlab_cli import api_service
from cleanlab_cli.types import (
    Schema,
    DataType,
    ValidationWarningType,
    RecordType,
    WarningLogType,
    RowWarningsType,
    FeatureType,
)
from cleanlab_cli.util import (
    is_null_value,
    dump_json,
    init_dataset_from_filepath,
    get_file_size,
)
from cleanlab_cli.dataset.schema_types import (
    PYTHON_TYPES_TO_READABLE_STRING,
    DATA_TYPES_TO_PYTHON_TYPES,
)
from cleanlab_cli import click_helpers
from cleanlab_cli.click_helpers import success, info, progress

VALIDATION_WARNING_TYPES: List[ValidationWarningType] = [
    "MISSING_ID",
    "MISSING_VAL",
    "TYPE_MISMATCH",
    "DUPLICATE_ID",
]


class ValidationWarning(Enum):
    MISSING_ID = 1
    MISSING_VAL = 2
    TYPE_MISMATCH = 3
    DUPLICATE_ID = 4


def warning_to_readable_name(warning: ValidationWarningType) -> str:
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


def get_value_type(val: Any) -> str:
    for python_type, readable_string in PYTHON_TYPES_TO_READABLE_STRING.items():
        if isinstance(val, python_type):
            return readable_string
    return "unrecognized"


def convert_to_python_type(val: Any, data_type: DataType) -> Any:
    return DATA_TYPES_TO_PYTHON_TYPES[data_type](val)


def validate_and_process_record(
    record: RecordType,
    schema: Schema,
    seen_ids: Set[str],
    existing_ids: Set[str],
    columns: Optional[List[str]] = None,
) -> Tuple[Optional[RecordType], Optional[str], Optional[RowWarningsType]]:
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
            {"MISSING_ID": [f"Missing ID for record: {dict(record)}."]},
        )

    row_id = str(row_id)
    if row_id in existing_ids:
        return None, row_id, None

    if row_id in seen_ids:
        return (
            None,
            row_id,
            {"DUPLICATE_ID": [f"Duplicate ID found: {dict(record)}"]},
        )

    warnings: Dict[ValidationWarningType, List[str]] = defaultdict(list)

    row = {c: record.get(c, None) for c in columns}

    for column_name, column_value in record.items():
        if column_name not in fields:
            continue
        col_type = fields[column_name]["data_type"]
        col_feature_type = fields[column_name]["feature_type"]

        warning: Optional[Tuple[str, ValidationWarningType]] = None
        if is_null_value(column_value):
            row[column_name] = None
            warning = f"{column_name}: value is missing", "MISSING_VAL"
        else:
            if col_feature_type == "datetime":
                try:
                    timestamp_value = convert_to_python_type(column_value, col_type)
                    pd.Timestamp(timestamp_value)
                except (ValueError, TypeError):
                    warning = (
                        f"{column_name}: expected datetime but unable to parse '{column_value}'"
                        f" with {get_value_type(column_value)} type. Datetime strings must be"
                        " parsable by pandas.Timestamp().",
                        "TYPE_MISMATCH",
                    )
            else:
                if col_type == "string":
                    row[column_name] = str(column_value)  # type coercion
                elif col_type == "integer":
                    if not isinstance(column_value, int):
                        if isinstance(column_value, str) and column_value.isdigit():
                            row[column_name] = int(column_value)
                        else:
                            warning = (
                                f"{column_name}: expected 'int' but got '{column_value}' with"
                                f" {get_value_type(column_value)} type",
                                "TYPE_MISMATCH",
                            )
                elif col_type == "float":
                    if isinstance(column_value, Decimal):
                        row[column_name] = float(column_value)
                    else:
                        if not (isinstance(column_value, int) or isinstance(column_value, float)):
                            coerced = False
                            if isinstance(column_value, str):
                                try:
                                    float_value = extract_float_string(column_value)
                                    row[column_name] = float(float_value)
                                    coerced = True
                                except Exception:
                                    pass
                            if not coerced:
                                warning = (
                                    f"{column_name}: expected 'float' but got '{column_value}' with"
                                    f" {get_value_type(column_value)} type",
                                    "TYPE_MISMATCH",
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
                                "TYPE_MISMATCH",
                            )

        if warning:
            row[column_name] = None  # replace bad value with NULL
            msg, warn_type = warning
            warnings[warn_type].append(msg)
    return row, row_id, warnings


def create_warning_log() -> WarningLogType:
    log: WarningLogType = {
        "MISSING_ID": [],
        "MISSING_VAL": dict(),
        "DUPLICATE_ID": dict(),
        "TYPE_MISMATCH": dict(),
    }
    return log


def update_log_with_warnings(
    log: WarningLogType, row_id: Optional[str], warnings: Optional[RowWarningsType]
) -> WarningLogType:
    if warnings:
        for warn_type in warnings:
            if warn_type == "MISSING_ID":
                log[warn_type] += warnings[warn_type]
            else:  # ID is present
                assert row_id is not None
                log[warn_type][row_id] = warnings[warn_type]
    return log


def echo_log_warnings(log: WarningLogType) -> None:
    for warning_type in VALIDATION_WARNING_TYPES:
        warning_count = len(log[warning_type])
        if warning_count > 0:
            click.echo(f"{warning_to_readable_name(warning_type)}: {warning_count}")


def validate_rows(
    dataset_filepath: str,
    columns: List[str],
    schema: Schema,
    log: WarningLogType,
    upload_queue: queue.Queue[Optional[List[Any]]],
    existing_ids: Optional[Collection[str]] = None,
) -> None:
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
        if row_id:
            seen_ids.add(row_id)

        if row:
            upload_queue.put(list(row.values()), block=True)

    upload_queue.put(None, block=True)


async def upload_rows(
    api_key: str,
    dataset_id: str,
    columns: List[str],
    upload_queue: queue.Queue[Optional[List[Any]]],
    rows_per_payload: int,
) -> None:
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
        upload_tasks: List[Union[Coroutine[Any, Any, None], Task[None]]] = []
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
    dataset_id: str,
    filepath: str,
    schema: Schema,
    existing_ids: Optional[Collection[str]] = None,
    output: Optional[str] = None,
    payload_size: float = 10,
) -> None:
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

    log = create_warning_log()

    file_size = get_file_size(filepath)
    api_service.check_dataset_limit(file_size, api_key=api_key, show_warning=False)

    # NOTE: makes simplifying assumption that first row size is representative of all row sizes
    row_size = getsizeof(next(init_dataset_from_filepath(filepath).read_streaming_records()))
    rows_per_payload = int(payload_size * 1e6 / row_size)
    upload_queue: queue.Queue[Optional[List[Any]]] = queue.Queue(maxsize=2 * rows_per_payload)

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

    total_warnings: int = sum([len(log[w]) for w in VALIDATION_WARNING_TYPES])
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
            save_warning_log(log, output)
            click_helpers.confirm_open_file(
                "Would you like to open your issues file for viewing?", filepath=output
            )
    click.secho(
        "Upload completed. View your uploaded dataset at https://app.cleanlab.ai",
        fg="green",
    )


def group_feature_types(schema: Schema) -> Dict[FeatureType, List[str]]:
    """
    Given a schema, return a dict mapping each feature type to the list of columns with said feature type
    """
    feature_types_to_columns = defaultdict(list)
    for field_name, spec in schema["fields"].items():
        feature_type = spec["feature_type"]
        feature_types_to_columns[feature_type].append(field_name)
    return feature_types_to_columns


def save_warning_log(warning_log: WarningLogType, filename: str) -> None:
    if not filename:
        raise ValueError("No filepath provided for saving warning_log")
    retval = {
        warning_to_readable_name(k): warning_log[k]
        for k in VALIDATION_WARNING_TYPES
        if k in warning_log
    }
    progress(f"Writing issues to {filename}...")
    dump_json(filename, retval)
    success("Saved.\n")


def extract_float_string(column_value: str) -> str:
    """
    Floating point: Decimal number containing a decimal point, optionally preceded by a + or - sign
    and optionally followed by the e or E character and a decimal number.

    Reference: https://docs.python.org/3/library/re.html#simulating-scanf
    """
    float_regex_pattern = r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?"
    float_value = re.search(float_regex_pattern, column_value)
    return float_value.group(0) if float_value else ""
