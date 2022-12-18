"""
Helper functions for working with schemas
"""
import decimal
import json
import os.path
import pathlib
import random
import re
from typing import Any, Collection, Dict, List, Optional, Sized, Tuple

import pandas as pd
import semver
from pandas import NaT

from cleanlab_studio.cli.click_helpers import abort, info, progress, success
from cleanlab_studio.internal.schema import DataType, FeatureType, Schema
from cleanlab_studio.internal.types import Modality
from cleanlab_studio.cli.util import dump_json, get_filename, init_dataset_from_filepath
from cleanlab_studio.version import MAX_SCHEMA_VERSION, MIN_SCHEMA_VERSION, SCHEMA_VERSION




def load_schema(filepath: str) -> Schema:
    with open(filepath, "r") as f:
        schema_dict = json.load(f)
        schema: Schema = Schema.create(
            metadata=schema_dict["metadata"],
            fields=schema_dict["fields"],
            version=schema_dict["version"],
        )
        return schema


def validate_schema(schema: Schema, columns: Collection[str]) -> None:
    """
    Checks that:
    (1) all schema column names are strings
    (2) all schema columns exist in the dataset columns
    (3) all schema column types are recognized

    Note that schema initialization already checks that all keys are present and that fields are valid.

    :param schema:
    :param columns: full list of columns in dataset
    :return: raises a ValueError if any checks fail
    """

    # check schema version validity
    schema_version = schema.version
    if (
        semver.VersionInfo.parse(MIN_SCHEMA_VERSION).compare(schema_version) == 1
    ):  # min schema > schema_version
        raise ValueError(
            "This schema version is incompatible with this version of the CLI. "
            "A new schema should be generated using 'cleanlab dataset schema generate'"
        )
    elif semver.VersionInfo.parse(MAX_SCHEMA_VERSION).compare(schema_version) == -1:
        raise ValueError(
            "CLI is not up to date with your schema version. Run 'pip install --upgrade cleanlab-studio'."
        )

    schema_columns = set(schema.fields)
    columns = set(columns)
    metadata = schema.metadata

    ## Check that the dataset has all columns specified in the schema
    if not schema_columns.issubset(columns):
        raise ValueError(f"Dataset is missing schema columns: {schema_columns - columns}")

    # Advanced validation checks: this should be aligned with ConfirmSchema's validate() function
    ## Check that specified ID column has the feature_type 'identifier'
    id_column_name = metadata.id_column
    id_column_spec_feature_type = schema.fields[id_column_name].feature_type
    if id_column_spec_feature_type != FeatureType.identifier:
        raise ValueError(
            f"ID column field {id_column_name} must have feature type: 'identifier', but has"
            f" feature type: '{id_column_spec_feature_type}'"
        )

    ## Check that there exists at least one categorical column (to be used as label)
    has_categorical = any(
        spec.feature_type == FeatureType.categorical for spec in schema.fields.values()
    )
    if not has_categorical:
        raise ValueError(
            "Dataset does not seem to contain a label column. (None of the fields is categorical.)"
        )

    ## If tabular modality, check that there are at least two variable (i.e. categorical, numeric, datetime) columns
    modality = metadata.modality
    variable_fields = {FeatureType.categorical, FeatureType.numeric, FeatureType.datetime}
    if modality == Modality.tabular:
        num_variable_columns = sum(
            int(spec.feature_type in variable_fields) for spec in schema.fields.values()
        )
        if num_variable_columns < 2:
            raise ValueError(
                "Dataset modality is tabular; there must be at least one categorical field and one"
                " other variable field (i.e. categorical, numeric, or datetime)."
            )

    ## If text modality, check that at least one column has feature type 'text'
    elif modality == Modality.text:
        has_text = any(spec.feature_type == FeatureType.text for spec in schema.fields.values())
        if not has_text:
            raise ValueError("Dataset modality is text, but none of the fields is a text column.")




def save_schema(schema: Schema, filename: Optional[str]) -> None:
    """

    :param schema:
    :param filename: filename to save schema with
    :return:
    """
    if filename == "":
        filename = "schema.json"
    if filename:
        progress(f"Writing schema to {filename}...")
        dump_json(filename, schema.to_dict())
        success("Saved.")
    else:
        info("Schema was not saved.")
