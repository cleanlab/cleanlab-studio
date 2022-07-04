"""
Contains utility functions for interacting with dataset files
"""
import json
import os
import pathlib
import jsonstreams
import pandas as pd

from cleanlab_cli.classes import CsvDataset, JsonDataset, ExcelDataset

ALLOWED_EXTENSIONS = [".csv", ".xls", ".xlsx", ".json"]


def get_file_extension(filename):
    file_extension = pathlib.Path(filename).suffix
    if file_extension in ALLOWED_EXTENSIONS:
        return file_extension
    raise ValueError(f"File extension for {filename} did not match allowed extensions.")


def is_allowed_extension(filename):
    return any([filename.endswith(ext) for ext in ALLOWED_EXTENSIONS])


def get_filename(filepath):
    return os.path.split(filepath)[-1]


def get_file_size(filepath):
    return os.path.getsize(filepath)


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


def init_dataset_from_filepath(filepath):
    ext = get_file_extension(filepath)
    if ext == ".csv":
        return CsvDataset(filepath)
    elif ext in [".xls", ".xlsx"]:
        return ExcelDataset(filepath)
    elif ext == ".json":
        return JsonDataset(filepath)


def dump_json(filepath, schema):
    with open(filepath, "w") as f:
        f.write(json.dumps(schema, indent=2))


def append_rows(rows, filename):
    df = pd.DataFrame(rows)
    extension = get_file_extension(filename)
    if extension == ".csv":
        if not os.path.exists(filename):
            df.to_csv(filename, mode="w", index=False, header=True)
        else:
            df.to_csv(filename, mode="a", index=False, header=False)
    elif extension in [".xls", ".xlsx"]:
        if not os.path.exists(filename):
            with pd.ExcelWriter(filename) as writer:
                df.to_excel(writer, index=False, header=True)
        else:
            with pd.ExcelWriter(filename) as writer:
                df.to_excel(writer, mode="a", index=False, header=False)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")


def get_dataset_chunks(filepath, id_column, ids_to_fields_to_values, num_rows_per_chunk):
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
    dataset_filepath,
    id_column,
    ids_to_fields_to_values,
    output_filepath,
    num_rows_per_chunk=10000,
):
    output_extension = get_file_extension(output_filepath)
    get_chunks = lambda: get_dataset_chunks(
        dataset_filepath, id_column, ids_to_fields_to_values, num_rows_per_chunk
    )

    if output_extension == ".json":
        with jsonstreams.Stream(
            jsonstreams.Type.OBJECT, filename=output_filepath, indent=True, pretty=True
        ) as s:
            with s.subarray("rows") as rows:
                for chunk in get_chunks():
                    for row in chunk:
                        rows.write(row)
    elif output_extension in [".csv", ".xls", ".xlsx"]:
        for chunk in get_chunks():
            append_rows(chunk, output_filepath)
    else:
        raise ValueError(f"Invalid file type: {output_extension}.")
