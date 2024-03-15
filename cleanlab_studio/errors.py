class InvalidUUIDError(ValueError):
    pass


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


class ValidationError(ValueError):
    pass


class VersionError(ValueError):
    pass


class MissingAPIKeyError(ValueError):
    pass


class APIError(Exception):
    pass


class APITimeoutError(APIError):
    pass


class RateLimitError(APIError):
    def __init__(self, message: str, retry_after: int):
        self.message = message
        self.retry_after = retry_after


class TlmBadRequest(APIError):
    pass


class TlmServerError(APIError):
    def __init__(self, message: str, status_code: int) -> None:
        self.message = message
        self.status_code = status_code


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
