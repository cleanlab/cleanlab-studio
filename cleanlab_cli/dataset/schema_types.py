from typing import Dict, Set

from cleanlab_cli.types import DataType, FeatureType

DATA_TYPES_TO_FEATURE_TYPES: Dict[DataType, Set[FeatureType]] = {
    "string": {"text", "categorical", "datetime", "identifier"},
    "integer": {"categorical", "datetime", "identifier", "numeric"},
    "float": {"datetime", "numeric"},
    "boolean": {"boolean"},
}

PYTHON_TYPES_TO_READABLE_STRING: Dict[type, DataType] = {
    str: "string",
    float: "float",
    int: "integer",
    bool: "boolean",
}

DATA_TYPES_TO_PYTHON_TYPES: Dict[DataType, type] = {
    "string": str,
    "float": float,
    "integer": int,
    "boolean": bool,
}
