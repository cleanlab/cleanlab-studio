"""
Contains utility functions for interacting with files and schemas
"""

import pandas as pd
from pandas.io.json import build_table_schema
from typing import Tuple, Optional, Iterable, Dict, List, Collection, Set
from random import sample, random
import pyexcel
import os
from collections import defaultdict
import pathlib
from config import SCHEMA_VERSION
from sqlalchemy import Integer, String, Boolean, DateTime, Float
import json
from api_service import *

ALLOWED_EXTENSIONS = ['.csv', '.xls', '.xlsx']

schema_mapper = {
    'integer': Integer(),
    'float': Float(),
    'string': String(),
    'boolean': Boolean(),
    'datetime': DateTime()
}

def preprocess_df(self):
    """
    Drop duplicate rows and columns with more than 20% of values missing,
    """
    df = self.dataframe
    len1 = len(df)
    df.drop_duplicates(inplace=True)
    len2 = len(df)

    cols_to_drop = []
    for c in df.columns:
        num_na = df[c].isna().sum()
        if num_na / len2 >= 0.2:  # drop column if it has more than 20% NA values
            cols_to_drop.append(c)

    df.drop(columns=cols_to_drop, inplace=True)

    df.dropna(inplace=True)
    len3 = len(df)
    num_dupe_rows = len1 - len2
    num_na_rows = len2 - len3
    return {
        'num_dupe_rows': num_dupe_rows,
        'num_na_rows': num_na_rows,
        'cols_dropped': cols_to_drop
    }


def get_df_column_type(df, col_name) -> Tuple[str, bool]:
    """
    Get data type of specified dataframe column
    """
    vals = list(df[col_name][~df[col_name].isna()])
    return infer_data_type(vals)


def get_file_extension(filename):
    file_extension = pathlib.Path(filename).suffix
    if file_extension in ALLOWED_EXTENSIONS:
        return file_extension
    raise ValueError(f"File extension for {filename} did not match allowed extensions.")


def is_allowed_extension(filename):
    return any([filename.endswith(ext) for ext in ALLOWED_EXTENSIONS])


def get_filename(filepath):
    return os.path.split(filepath)[-1]

def infer_data_type(vals) -> Tuple[str, bool]:
    """
    Infer the data type of a collection of a values using simple heuristics.

    :param vals: an iterable containing data values
    :return: (text, categorical, or numeric), bool for whether column is possibly an id
    """
    counts = {
        'string': 0,
        'numeric': 0
    }
    sample_size = 100

    ratio_unique = len(set(vals)) / len(vals)

    id_like = ratio_unique >= 0.90  # accounting for datasets with duplicate IDs

    for x in sample(vals, sample_size):
        if isinstance(x, str):
            counts['string'] += 1
        elif isinstance(x, float) or isinstance(x, int):
            counts['numeric'] += 1

    if counts['string'] >= 10 and counts['numeric'] >= 10:
        return "string", id_like
    elif counts['numeric'] > 0:
        return 'numeric', id_like
    elif counts['string'] > 0:
        if ratio_unique < 0.2:  # heuristic: there should be on average >=5 examples for each string example
            return 'categorical', id_like
        else:
            return 'text', id_like


def read_file_as_df(filepath, filetype):
    if filetype == '.json':
        df = pd.read_json(filepath, convert_axes=False, convert_dates=False).T
        df.index = df.index.astype('str')
        df['id'] = df.index
    elif filetype in '.csv':
        df = pd.read_csv(filepath)
    elif filetype in ['.xls', '.xlsx']:
        df = pd.read_excel(filepath)
    elif filetype in ['.parquet']:
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Failed to read filetype: {filetype}")
    return df


def extract_details(self):
    df = self.dataframe
    stats = self.preprocess_details
    cols = list(df.columns)

    id_column = self._find_best_matching_column('id', cols)
    text_column = self._find_best_matching_column('text', cols)
    label_column = self._find_best_matching_column('label', cols)

    special_columns = [id_column, text_column, label_column]
    special_columns = [c for c in special_columns if c is not None]

    numeric_cols = []
    categorical_cols = []
    possible_id_cols = []
    possible_text_cols = []

    for col in cols:
        column_type, id_like = get_df_column_type(df, col)
        if id_like:
            possible_id_cols.append(col)
        if column_type == 'numeric':
            numeric_cols.append(col)
        elif column_type == 'categorical':
            categorical_cols.append(col)
        elif column_type == 'text':
            possible_text_cols.append(col)

    possible_feature_cols = numeric_cols + categorical_cols

    used_columns = set(possible_feature_cols + special_columns)
    unused_columns = list(set(cols) - used_columns)

    retval = {
        'num_rows': len(df),
        'dupe_rows': stats['num_dupe_rows'],
        'cols_dropped': stats['cols_dropped'],
        'na_rows': stats['num_na_rows'],
        'filetype': self.filetype,
        'filename': self.filename,
        'id_col': id_column,
        'text_col': text_column,
        'label_col': label_column,
        'cols': cols,
        'possible_id_cols': possible_id_cols,
        'possible_feature_cols': possible_feature_cols,
        'possible_label_cols': categorical_cols,
        'possible_text_cols': possible_text_cols,
        'unused_cols': unused_columns
    }
    return retval

def is_null_value(val):
    return val is None or val == ''


def diagnose_dataset(filepath: str, threshold: float = 0.2):
    """
    Generates an initial diagnostic for the dataset before any pre-processing and type validation.

    The diagnostic consists of: (1) a list of columns with >=20% null values, (2) the list of row IDs with null values,
    (3) the total number of rows in the dataset

    Throws a KeyError if the `id_col` does not exist in the dataset.

    :param filepath:
    :param threshold:
    :return:
    """

    stream = read_file_as_stream(filepath)
    num_rows = 0
    col_to_null_count = defaultdict(int)
    for row in stream:
        num_rows += 1
        for k, v in row.items():
            if is_null_value(v):
                col_to_null_count[k] += 1

    null_cols = [col for col, count in col_to_null_count.items() if count / num_rows >= threshold]
    return null_cols, num_rows


def convert_schema_to_dtypes(schema):
    dtypes = {}
    # version = schema['version']
    for field in schema['fields']:
        dtypes[field['name']] = schema_mapper[field['value'].lower()]
    return dtypes


def load_schema(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def validate_schema_fields(schema, columns: Set[str]):
    """
    Checks that:
    (1) all schema column names are strings
    (2) the schema is not missing any kept columns in the dataset
    (3) all schema column types are recognized

    :param schema:
    :param columns:
    :return: raises a ValueError if any checks fail
    """
    schema_columns = set(schema['fields'])
    for col in schema_columns:
        if not isinstance(col, str):
            raise ValueError(f"All schema columns must be strings. Found invalid column name: {col}")

    if not columns.issubset(schema_columns):
        raise ValueError(f"Schema is missing columns: {columns - schema_columns}")

    for column_type in schema['fields'].values():
        if column_type not in schema_mapper:
            raise ValueError(f"Unrecognized column data type: {column_type}")


def propose_schema(filepath: str, columns: Collection[str], num_rows: int, sample_size: int = 1000) -> Dict[str, str]:
    """
    Generates a schema for a dataset based on a sample of up to 1000 of the dataset's rows.
    :param filepath:
    :param columns: columns to generate a schema for
    :param num_rows: number of rows in dataset
    :param sample_size:
    :return:

    """
    stream = read_file_as_stream(filepath)
    dataset = []
    sample_proba = 1 if sample_size >= num_rows else sample_size / num_rows
    for row in stream:
        if random() <= sample_proba:
            dataset.append(dict(row.items()))
    df = pd.DataFrame(dataset, columns=columns)
    schema = build_table_schema(df)
    retval = {}
    retval['fields'] = {}
    for entry in schema['fields']:
        column_type = entry['type']
        if column_type == 'number':
            column_type = 'float'
        retval['fields'][entry['name']] = column_type
    retval['version'] = SCHEMA_VERSION
    return retval


def get_dataset_columns(filepath):
    stream = read_file_as_stream(filepath)
    for r in stream:
        return list(r.keys())


def read_file_as_stream(filepath):
    """
    Opens a file and reads it as a stream (aka row-by-row) to limit memory usage
    :param filepath: path to target file
    :return: a generator that yields dataset rows, with each row being an OrderedDict
    """
    ext = get_file_extension(filepath)

    if ext in ['.csv', '.xls', '.xlsx']:
        for r in pyexcel.iget_records(file_name=filepath):
            yield r

def validate_row(row, schema):
    pass

def upload_rows(filepath, schema):
    """

    :param filepath: path to dataset file
    :param schema: a validated schema
    :return: None
    """
    for r in read_file_as_stream(filepath):
        pass

