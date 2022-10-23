import jsonstreams
from typing import Dict

from cleanlab_studio.cli.types import RecordType, DatasetFileExtension
from cleanlab_studio.cli.util import get_dataset_file_extension, get_dataset_chunks, append_rows


def combine_fields_with_dataset(
    dataset_filepath: str,
    id_column: str,
    ids_to_fields_to_values: Dict[str, RecordType],
    output_filepath: str,
    num_rows_per_chunk: int = 10000,
) -> None:
    output_extension = get_dataset_file_extension(output_filepath)
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
