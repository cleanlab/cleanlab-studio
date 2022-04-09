from sqlalchemy import String, Boolean, DateTime, Float, BigInteger

from setup import VERSION_NO

schema_mapper = {
    "string": String(),
    "integer": BigInteger(),
    "float": Float(),
    "boolean": Boolean(),
    "datetime": DateTime(),
}

DATA_TYPES_TO_FEATURE_TYPES = {
    "string": {"text", "categorical", "datetime", "identifier"},
    "integer": {"categorical", "datetime", "identifier", "numeric"},
    "float": {"datetime", "numeric"},
    "boolean": {"boolean"},
}

PYTHON_TYPES_TO_READABLE_STRING = {str: "string", float: "float", int: "integer", bool: "boolean"}

SCHEMA_VERSION = VERSION_NO
