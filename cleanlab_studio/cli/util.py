"""
Contains utility functions for interacting with dataset files
"""
import json
import os
import pathlib
import pandas as pd
from typing import Optional, Dict, Any, List, Generator

from cleanlab_studio.cli.classes import CsvDataset, JsonDataset, ExcelDataset
from cleanlab_studio.cli.classes.dataset import Dataset
from cleanlab_studio.cli.types import (
    RecordType,
    DatasetFileExtension,
    ImageFileExtension,
)


def get_dataset_file_extension(filename: str) -> DatasetFileExtension:
    file_extension = pathlib.Path(filename).suffix
    return DatasetFileExtension(file_extension)


def _standardize_image_file_extension(file_extension: str) -> str:
    file_extension = file_extension.lower()
    if file_extension == ".jpg":
        return ".jpeg"
    return file_extension


def get_image_file_extension(filename: str) -> ImageFileExtension:
    file_extension = pathlib.Path(filename).suffix
    return ImageFileExtension(_standardize_image_file_extension(file_extension))


def get_filename(filepath: str) -> str:
    return os.path.split(filepath)[-1]


def get_file_size(filepath: str, ignore_missing_files: bool = False) -> int:
    if ignore_missing_files:
        try:
            return os.path.getsize(filepath)
        except IOError:
            return 0
    return os.path.getsize(filepath)


def read_file_as_df(filepath: str) -> pd.DataFrame:
    ext = get_dataset_file_extension(filepath)
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
    ext = get_dataset_file_extension(filepath)
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
    ext = get_dataset_file_extension(filename)
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
            with pd.ExcelWriter(filename, mode="a") as writer:
                df.to_excel(writer, index=False, header=False)


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
