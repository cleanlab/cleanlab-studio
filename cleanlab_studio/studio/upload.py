"""
Utilities to upload datasets.
"""

# This implementation overlaps with the one in
# cleanlab_studio.cli.upload_helpers. The two should be unified and put in
# cleanlab_studio.internal.

import random
import json
import gzip
import requests
import io
from typing import Optional
import multiprocessing.dummy

from tqdm.auto import tqdm
import pandas as pd

from cleanlab_studio.internal import api
from cleanlab_studio.internal.dataset import PandasDataset
from cleanlab_studio.internal.schema import Schema
from cleanlab_studio.cli.dataset.upload_helpers import validate_and_process_record
from cleanlab_studio.version import SCHEMA_VERSION

IMAGE_UPLOAD_CHECKPOINT_SIZE = 100
MAX_PARALLEL_UPLOADS = 32


def upload_tabular_dataset(
    api_key: str,
    dataset: PandasDataset,
    schema: Optional[Schema] = None,
    dataset_id: Optional[str] = None,
) -> str:
    if dataset_id is None:
        # if ID is not specified, initialize dataset
        assert schema is not None
        dataset_id = api.initialize_dataset(api_key, schema)
        seen = set()
    else:
        # resuming upload
        if schema is None:
            schema = api.get_dataset_schema(api_key, dataset_id)
        seen = set(api.get_existing_ids(api_key, dataset_id))

    # extract rows from dataset, according to schema, filtering out existing IDs
    fields = schema.fields
    id_column = schema.metadata.id_column
    columns = list(fields)
    to_upload = []
    for record in dataset.read_streaming_records():
        # XXX using CLI function here
        row, row_id, warnings = validate_and_process_record(dataset, record, schema, seen, seen)
        if row is None or row[id_column] in seen:
            continue
        seen.add(row[id_column])
        to_upload.append(list(row.values()))  # ordered dict, order is preserved

    # split upload into chunks
    # estimate size per row
    nelem = min(len(to_upload), 10)
    size_per_row = (
        len(gzip.compress(json.dumps(random.sample(to_upload, nelem)).encode("utf8"))) / nelem
    )
    num_per_chunk = max(int(10 * 1024 * 1024 / size_per_row), 1)

    chunks = split_into_chunks(to_upload, num_per_chunk)

    # upload all chunks (no parallelism here)
    with tqdm(total=len(to_upload), unit="row") as pbar:
        for chunk in chunks:
            api.upload_rows(api_key, dataset_id, chunk, columns)
            pbar.update(len(chunk))

    api.complete_upload(api_key, dataset_id)

    return dataset_id


def get_type(schema, path):
    while "." in path:
        first, path = path.split(".", 1)
        schema = schema[first].dataType
    return schema[path].dataType


# dataset here is a pyspark DataFrame
def upload_image_dataset(
    api_key: str,
    dataframe,
    name: str,
    id_column: str,
    path_column: str,
    content_column: str,
    label_column: str,
    dataset_id: Optional[str] = None,
) -> str:
    spark = dataframe.sparkSession
    if get_type(dataframe.schema, content_column).typeName() != "binary":
        raise ValueError("content column must have binary type")
    if dataset_id is None:
        # if ID is not specified, initialize dataset
        # create an appropriate schema, like we do in simple_image_upload
        label_column_type = get_type(dataframe.schema, label_column).typeName()
        # note: 'string' and 'integer' are DataType values; spark names line up with our internal names
        if label_column_type not in ["string", "integer"]:
            raise ValueError("label column must have string or integer type")
        schema = Schema.create(
            metadata={
                "id_column": id_column,
                "modality": "image",
                "name": name,
                "filepath_column": path_column,
            },
            fields={
                id_column: {"data_type": "string", "feature_type": "identifier"},
                path_column: {"data_type": "string", "feature_type": "filepath"},
                label_column: {"data_type": label_column_type, "feature_type": "categorical"},
            },
            version=SCHEMA_VERSION,
        )
        dataset_id = api.initialize_dataset(api_key, schema)
        seen = set()
    else:
        # resuming upload
        if schema is None:
            schema = api.get_dataset_schema(api_key, dataset_id)
        seen = set(api.get_existing_ids(api_key, dataset_id))
    fields = schema.fields
    columns = list(fields)

    metadata_df = dataframe.select([id_column, path_column, label_column]).toPandas()
    to_upload = []
    for idx, row in metadata_df.iterrows():
        row = row.tolist()
        row_id = row[0]
        if row_id not in seen:
            to_upload.append(row)
            seen.add(row_id)

    chunks = split_into_chunks(to_upload, IMAGE_UPLOAD_CHECKPOINT_SIZE)

    # upload chunks (each chunk in serial, with images uploaded in parallel)
    with tqdm(total=len(to_upload), unit="row") as pbar:
        for chunk in chunks:
            row_ids = [row[0] for row in chunk]
            filepaths = [row[1] for row in chunk]
            filepath_to_post = api.get_presigned_posts(
                api_key, dataset_id, filepaths=filepaths, row_ids=row_ids, media_type="image"
            )
            # get contents for this chunk at once
            ids_df = spark.createDataFrame(pd.DataFrame({id_column: row_ids}))
            contents = {
                row[0]: row[1]
                for row in ids_df.join(dataframe, on=id_column, how="left")
                .select([id_column, content_column])
                .collect()
            }

            def upload_row(row):
                image_file = io.BytesIO(contents[row[0]])
                post_data = filepath_to_post[row[1]]  # indexed by filepath
                presigned_post = post_data["post"]
                res = requests.post(
                    url=presigned_post["url"],
                    data=presigned_post["fields"],
                    files={"file": image_file},
                )
                if not res.ok:
                    raise Exception(f"failure while uploading id {row[0]}")

            with multiprocessing.dummy.Pool(MAX_PARALLEL_UPLOADS) as p:
                # thread pool, not process pool; requests releases GIL in post
                p.map(upload_row, chunk)

            # upload metadata
            api.upload_rows(api_key, dataset_id, chunk, columns)
            pbar.update(len(chunk))

    api.complete_upload(api_key, dataset_id)

    return dataset_id


def split_into_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]
