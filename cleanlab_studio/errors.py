class InvalidDatasetError(ValueError):
    pass


class EmptyDatasetError(InvalidDatasetError):
    pass


class ColumnMismatchError(InvalidDatasetError):
    pass


class InvalidSchemaError(InvalidDatasetError):
    pass


class MissingPathError(ValueError):
    pass


class NotInstalledError(ImportError):
    pass


class SettingsError(ValueError):
    pass


class UploadError(ValueError):
    pass


class VersionError(ValueError):
    pass


class MissingAPIKeyError(ValueError):
    pass


class APIError(Exception):
    pass


class IngestionError(APIError):
    def __init__(self, error_type: str, message: str) -> None:
        self.error_type = error_type
        self.message = message

    def __str__(self) -> str:
        return f"{self.error_type}: {self.message}"


class APITimeoutError(APIError):
    pass


class RateLimitError(APIError):
    def __init__(self, message: str, retry_after: int):
        self.message = message
        self.retry_after = retry_after


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


class InvalidSchemaTypeError(ValueError):
    def __init__(self, msg: str) -> None:
        self.msg = msg

    def __str__(self) -> str:
        return f"{self.msg}\nSee [TODO: insert docs] link for more information."
