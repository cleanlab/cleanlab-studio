from decimal import Decimal
from config import __version__

DATA_TYPES_TO_FEATURE_TYPES = {
    "string": {"text", "categorical", "datetime", "identifier"},
    "integer": {"categorical", "datetime", "identifier", "numeric"},
    "float": {"datetime", "numeric"},
    "boolean": {"boolean"},
}

PYTHON_TYPES_TO_READABLE_STRING = {
    str: "string",
    float: "float",
    int: "integer",
    bool: "boolean",
    Decimal: "float",
}

SCHEMA_VERSION = __version__
