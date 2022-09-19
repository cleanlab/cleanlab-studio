"""
Contains utility functions for interacting with dataset files
"""
import json
import os
import pathlib
import jsonstreams
import pandas as pd
from typing import Optional, Dict, Any, List, Generator

from cleanlab_cli.classes import CsvDataset, JsonDataset, ExcelDataset
from cleanlab_cli.classes.dataset import Dataset
from cleanlab_cli.types import (
    RecordType,
    DatasetFileExtension,
)


def get_file_extension(filename: str) -> DatasetFileExtension:
    file_extension = pathlib.Path(filename).suffix
    return DatasetFileExtension(file_extension)


def get_filename(filepath: str) -> str:
    return os.path.split(filepath)[-1]


def get_file_size(filepath: str) -> int:
    return os.path.getsize(filepath)


def read_file_as_df(filepath: str) -> pd.DataFrame:
    ext = get_file_extension(filepath)
    if ext == DatasetFileExtension.json:
        df = pd.read_json(filepath, convert_axes=False, convert_dates=False).T
        df.index = df.index.astype("str")
        df["id"] = df.index
    elif ext == DatasetFileExtension.csv:
        df = pd.read_csv(filepath, keep_default_na=True)
    else:  # ext == DatasetFileExtension.xls or ext == DatasetFileExtension.xlsx:
        df = pd.read_excel(filepath, keep_default_na=True)
    # elif ext in [".parquet"]:
    #     df = pd.read_parquet(filepath)
    return df


def is_null_value(val: str) -> bool:
    return val is None or val == "" or pd.isna(val)


def init_dataset_from_filepath(filepath: str) -> Dataset:
    ext = get_file_extension(filepath)
    if ext == DatasetFileExtension.csv:
        return CsvDataset(filepath)
    elif ext == DatasetFileExtension.xls or ext == DatasetFileExtension.xlsx:
        return ExcelDataset(filepath)
    else:  # ext == ".json":
        return JsonDataset(filepath)


def dump_json(filepath: str, obj: object) -> None:
    with open(filepath, "w") as f:
        f.write(json.dumps(obj, indent=2))


def append_rows(rows: List[RecordType], filename: str) -> None:
    df = pd.DataFrame(rows)
    ext = get_file_extension(filename)
    if ext == DatasetFileExtension.csv:
        if not os.path.exists(filename):
            df.to_csv(filename, mode="w", index=False, header=True)
        else:
            df.to_csv(filename, mode="a", index=False, header=False)
    elif ext == DatasetFileExtension.xls or ext == DatasetFileExtension.xlsx:
        if not os.path.exists(filename):
            with pd.ExcelWriter(filename) as writer:
                df.to_excel(writer, index=False, header=True)
        else:
            with pd.ExcelWriter(filename) as writer:
                df.to_excel(writer, mode="a", index=False, header=False)


def get_dataset_chunks(
    filepath: str,
    id_column: str,
    ids_to_fields_to_values: Dict[str, RecordType],
    num_rows_per_chunk: int,
) -> Generator[List[RecordType], None, None]:
    dataset = init_dataset_from_filepath(filepath)
    chunk = []
    for r in dataset.read_streaming_records():
        row_id = str(r.get(id_column))
        if row_id:
            updates = ids_to_fields_to_values[row_id]
            r.update(updates)
        chunk.append(r)

        if len(chunk) >= num_rows_per_chunk:
            yield chunk
            chunk = []

    if chunk:
        yield chunk


def combine_fields_with_dataset(
    dataset_filepath: str,
    id_column: str,
    ids_to_fields_to_values: Dict[str, RecordType],
    output_filepath: str,
    num_rows_per_chunk: int = 10000,
) -> None:
    output_extension = get_file_extension(output_filepath)
    if output_extension == DatasetFileExtension.json:
        with jsonstreams.Stream(
            jsonstreams.Type.OBJECT, filename=output_filepath, indent=True, pretty=True
        ) as s:
            with s.subarray("rows") as rows:
                for chunk in get_dataset_chunks(
                    dataset_filepath, id_column, ids_to_fields_to_values, num_rows_per_chunk
                ):
                    for row in chunk:
                        rows.write(row)
    elif output_extension in [
        DatasetFileExtension.csv,
        DatasetFileExtension.xls,
        DatasetFileExtension.xlsx,
    ]:
        for chunk in get_dataset_chunks(
            dataset_filepath, id_column, ids_to_fields_to_values, num_rows_per_chunk
        ):
            append_rows(chunk, output_filepath)
    else:
        raise ValueError(f"Invalid file type: {output_extension}.")
