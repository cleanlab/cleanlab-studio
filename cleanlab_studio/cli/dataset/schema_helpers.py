"""
Helper functions for working with schemas
"""

import json
from typing import Any, cast, List

from cleanlab_studio.internal.types import JSONDict, SchemaOverride


def load_schema_overrides(filepath: str) -> List[SchemaOverride]:
    with open(filepath, "r") as f:
        schema_overrides: Any = json.load(f)
        _validate_schema_overrides(schema_overrides)
        return cast(List[SchemaOverride], schema_overrides)


def _validate_schema_overrides(schema_overrides: List[JSONDict]) -> None:
    """
    Checks that schema overrides are formed as a list of JSON objects, with each object
    containing the following keys:
        - name
        - column_type
    """
    assert isinstance(schema_overrides, list), "schema_overrides must be a list of JSON objects"
    assert all(
        isinstance(schema_override, dict) for schema_override in schema_overrides
    ), "schema_overrides must be a list of JSON objects"
    assert all(
        "name" in schema_override and "column_type" in schema_override
        for schema_override in schema_overrides
    ), "each schema overrides must have a 'name' and 'column_type' key"
