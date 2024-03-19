class HandledError(Exception):
    pass


class InvalidUUIDError(HandledError):
    pass


class InvalidDatasetError(HandledError):
    pass


class EmptyDatasetError(HandledError):
    pass


class ColumnMismatchError(HandledError):
    pass


class InvalidSchemaError(HandledError):
    pass


class MissingPathError(HandledError):
    pass


class NotInstalledError(HandledError):
    pass


class SettingsError(HandledError):
    pass


class UploadError(HandledError):
    pass


class VersionError(HandledError):
    pass


class MissingAPIKeyError(HandledError):
    pass


class APIError(Exception):
    pass


class IngestionError(APIError):
    def __init__(self, error_type: str, message: str) -> None:
        self.error_type = error_type
        self.message = message

    def __str__(self) -> str:
        return f"{self.error_type}: {self.message}"


class APITimeoutError(HandledError):
    pass


class RateLimitError(APIError):
    def __init__(self, message: str, retry_after: int):
        self.message = message
        self.retry_after = retry_after


class TlmBadRequest(APIError):
    pass


class UnsupportedVersionError(HandledError):
    def __init__(self) -> None:
        super().__init__(
            "cleanlab-studio is out of date and must be upgraded. Run 'pip install --upgrade cleanlab-studio'."
        )


class AuthError(HandledError):
    def __init__(self) -> None:
        super().__init__("invalid API key")


class InternalError(HandledError):
    pass


class CleansetError(InternalError):
    pass


class InvalidSchemaTypeError(ValueError):
    def __init__(self, msg: str) -> None:
        self.msg = msg

    def __str__(self) -> str:
        return f"{self.msg}\nSee [TODO: insert docs] link for more information."
