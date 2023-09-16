class InvalidDatasetError(ValueError):
    pass


class EmptyDatasetError(InvalidDatasetError):
    pass


class ColumnMismatchError(InvalidDatasetError):
    pass


class APIError(Exception):
    pass


class UnsupportedVersionError(APIError):
    def __init__(self) -> None:
        super().__init__(
            "cleanlab-studio is out of date and must be upgraded. Run 'pip install --upgrade cleanlab-studio'."
        )


class AuthError(APIError):
    def __init__(self) -> None:
        super().__init__("invalid API key")


class InternalError(Exception):
    pass


class CleansetError(InternalError):
    pass
