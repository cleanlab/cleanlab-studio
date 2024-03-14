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
