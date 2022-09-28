"""
Helper functions for processing and uploading dataset rows
"""
import asyncio
import os.path
import threading
import queue
from asyncio import Task

import aiohttp
import click
import json
import pandas as pd
import re
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

from tqdm import tqdm

from cleanlab_cli import api_service
from cleanlab_cli.classes.dataset import Dataset
from cleanlab_cli.dataset.image_utils import is_valid_image
from cleanlab_cli.dataset.upload_types import (
    ValidationWarning,
    WarningLog,
    RowWarningsType,
    warning_to_readable_name,
)
from cleanlab_cli.types import (
    RecordType,
    Modality,
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
    DataType,
    Schema,
    FeatureType,
)
from cleanlab_cli import click_helpers
from cleanlab_cli.click_helpers import success, info, progress, abort


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
    :param existing_ids:
    :return: tuple (processed row: dict[str, any], row ID: optional[str], warnings: dict[warn_type: str, desc: str])
    """
    fields = schema.fields
    id_column = schema.metadata.id_column
    columns = list(fields)

    row_id = record.get(id_column, None)

    if row_id == "" or row_id is None:
        return (
            None,
            None,
            {ValidationWarning.MISSING_ID: [f"Missing ID for record: {dict(record)}."]},
        )

    row_id = str(row_id)
    if row_id in existing_ids:
        return None, row_id, None

    if row_id in seen_ids:
        return (
            None,
            row_id,
            {ValidationWarning.DUPLICATE_ID: [f"Duplicate ID found: {dict(record)}"]},
        )

    warnings: Dict[ValidationWarning, List[str]] = defaultdict(list)

    row = {c: record.get(c, None) for c in columns}

    for column_name, column_value in record.items():
        if column_name not in fields:
            continue
        col_type = fields[column_name].data_type
        col_feature_type = fields[column_name].feature_type

        warning: Optional[Tuple[str, ValidationWarning]] = None
        if is_null_value(column_value):
            row[column_name] = None
            warning = f"{column_name}: value is missing", ValidationWarning.MISSING_VAL
        else:
            if col_feature_type == FeatureType.datetime:
                try:
                    timestamp_value = convert_to_python_type(column_value, col_type)
                    pd.Timestamp(timestamp_value)
                except (ValueError, TypeError):
                    warning = (
                        f"{column_name}: expected datetime but unable to parse '{column_value}'"
                        f" with {get_value_type(column_value)} type. Datetime strings must be"
                        " parsable by pandas.Timestamp().",
                        ValidationWarning.TYPE_MISMATCH,
                    )
            elif col_feature_type == FeatureType.filepath:
                if schema.metadata.modality == Modality.image:
                    if not os.path.exists(column_value):
                        warning = (
                            f"{column_name}: expected filepath but unable to find file at specified filepath {column_value}. "
                            f"Filepath must be absolute or relative to your current working directory: {os.getcwd()}",
                            ValidationWarning.MISSING_FILE,
                        )
                    else:
                        if not is_valid_image(column_value):
                            warning = (
                                f"{column_name}: could not open invalid file at {column_value}. Image file must have",
                                ValidationWarning.INVALID_FILE,
                            )

            else:
                if col_type == DataType.string:
                    row[column_name] = str(column_value)  # type coercion
                elif col_type == DataType.integer:
                    if not isinstance(column_value, int):
                        if isinstance(column_value, str) and column_value.isdigit():
                            row[column_name] = int(column_value)
                        else:
                            warning = (
                                f"{column_name}: expected 'int' but got '{column_value}' with"
                                f" {get_value_type(column_value)} type",
                                ValidationWarning.TYPE_MISMATCH,
                            )
                elif col_type == DataType.float:
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
                                ValidationWarning.TYPE_MISMATCH,
                            )
                elif col_type == DataType.boolean:
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
            warnings[warn_type].append(msg)
    return row, row_id, warnings


def create_warning_log() -> WarningLog:
    return WarningLog.init()


def update_log_with_warnings(
    log: WarningLog, row_id: Optional[str], warnings: Optional[RowWarningsType]
) -> WarningLog:
    if warnings:
        for warn_type in warnings:
            if warn_type == ValidationWarning.MISSING_ID:
                log_warnings = log.get(warn_type)
                assert isinstance(log_warnings, list)
                log_warnings += warnings[warn_type]
            else:  # ID is present
                assert row_id is not None
                log_warnings = log.get(warn_type)
                assert isinstance(log_warnings, dict)
                log_warnings[row_id] = warnings[warn_type]
    return log


def echo_log_warnings(log: WarningLog) -> None:
    for warning_type in ValidationWarning:
        warning_count = len(log.get(warning_type))
        if warning_count > 0:
            click.echo(f"{warning_to_readable_name(warning_type)}: {warning_count}")


def validate_rows(
    dataset_filepath: str,
    schema: Schema,
    log: WarningLog,
    upload_queue: "queue.Queue[Optional[List[Any]]]",
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

    process_dataset(
        dataset=dataset,
        schema=schema,
        seen_ids=seen_ids,
        existing_ids=existing_ids,
        log=log,
        upload_queue=upload_queue,
    )
    upload_queue.put(None, block=True)


async def upload_rows(
    api_key: str,
    dataset_id: str,
    columns: List[str],
    upload_queue: "queue.Queue[Optional[List[Any]]]",
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


def check_filepath_column(modality: Modality, dataset_filepath: str, filepath_column: str) -> None:
    """
    Check the filepath column of a dataset to see if any of the filepaths are invalid.
    If >0 filepaths are invalid, print the number of invalid filepaths and prompt user for confirmation about
    outputting filepaths to console.
    """
    dataset = init_dataset_from_filepath(dataset_filepath)
    if filepath_column not in dataset.get_columns():
        raise ValueError(
            f"No filepath column '{filepath_column}' found in dataset at {dataset_filepath}."
        )
    nonexistent_filepaths = []
    for record in dataset.read_streaming_records():
        filepath_value = record[filepath_column]
        if not os.path.exists(filepath_value):
            nonexistent_filepaths.append(filepath_value)

    num_nonexistent_filepaths = len(nonexistent_filepaths)
    if num_nonexistent_filepaths > 0:
        click.echo(
            f"Found {num_nonexistent_filepaths} non-existent filepaths in specified filepath column: {filepath_column}. "
            f"As this is a {modality.value} dataset, these {num_nonexistent_filepaths} rows will be skipped unless the filepaths are fixed. "
            f"Filepaths must be absolute or relative to your current working directory: {os.getcwd()}"
        )
        to_print = click.confirm(
            f"Would you like to print the {num_nonexistent_filepaths} filepaths to console?"
        )
        if to_print:
            click.echo("Invalid filepaths:\n")
            click.echo(", ".join(nonexistent_filepaths))

        continue_with_upload = click.confirm(
            "Would you like to proceed with dataset upload? Rows with invalid filepaths will be skipped."
        )
        if not continue_with_upload:
            abort("Dataset upload aborted.")


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
    columns = list(schema.fields.keys())
    log = create_warning_log()

    file_size = get_file_size(filepath)
    api_service.check_dataset_limit(file_size, api_key=api_key, show_warning=False)

    if schema.metadata.modality == Modality.image:
        assert schema.metadata.filepath_column is not None
        check_filepath_column(
            modality=schema.metadata.modality,
            dataset_filepath=filepath,
            filepath_column=schema.metadata.filepath_column,
        )

    # NOTE: makes simplifying assumption that first row size is representative of all row sizes
    row_size = getsizeof(next(init_dataset_from_filepath(filepath).read_streaming_records()))
    rows_per_payload = int(payload_size * 1e6 / row_size)
    upload_queue: queue.Queue[Optional[List[Any]]] = queue.Queue(maxsize=2 * rows_per_payload)

    # create validation process
    validation_thread = threading.Thread(
        target=validate_rows,
        kwargs={
            "dataset_filepath": filepath,
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

    total_warnings: int = sum([len(log.get(w)) for w in ValidationWarning])
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
    for field_name, spec in schema.fields.items():
        feature_types_to_columns[spec.feature_type].append(field_name)
    return feature_types_to_columns


def save_warning_log(warning_log: WarningLog, filename: str) -> None:
    if not filename:
        raise ValueError("No filepath provided for saving warning_log")
    progress(f"Writing issues to {filename}...")
    dump_json(filename, warning_log.to_dict(readable=True))
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


def process_dataset(
    dataset: Dataset,
    schema: Schema,
    seen_ids: Set[str],
    existing_ids: Set[str],
    log: WarningLog,
    upload_queue: Optional["queue.Queue[Optional[List[Any]]]"] = None,
) -> None:
    """
    Validate and processes records, while updating the warning log
    If an upload queue is provided, valid rows are put on it
    """
    for record in tqdm(
        dataset.read_streaming_records(), total=len(dataset), initial=1, leave=True, unit=" rows"
    ):
        row, row_id, warnings = validate_and_process_record(record, schema, seen_ids, existing_ids)
        update_log_with_warnings(log, row_id, warnings)
        # row and row ID both present, i.e. row will be uploaded
        if row_id:
            seen_ids.add(row_id)

        if upload_queue and row is not None:
            upload_queue.put(list(row.values()), block=True)
