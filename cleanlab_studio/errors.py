import pathlib
from asyncio import Handle
from typing import Union


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


class ValidationError(HandledError):
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


class RateLimitError(HandledError):
    def __init__(self, message: str, retry_after: int):
        self.message = message
        self.retry_after = retry_after


class TlmBadRequest(HandledError):
    pass


class TlmServerError(APIError):
    def __init__(self, message: str, status_code: int) -> None:
        self.message = message
        self.status_code = status_code


class TlmPartialSuccess(APIError):
    """TLM request partially succeeded. Still returns result to user."""

    pass


class UnsupportedVersionError(HandledError):
    def __init__(self) -> None:
        super().__init__(
            "cleanlab-studio is out of date and must be upgraded. Run 'pip install --upgrade cleanlab-studio'."
        )


class AuthError(HandledError):
    def __init__(self) -> None:
        super().__init__(
            "API key is invalid. Check https://app.cleanlab.ai/upload for your current API key."
        )


class InternalError(HandledError):
    pass


class CleansetError(InternalError):
    pass


class CleansetHandledError(InternalError):
    DEFAULT_ERROR_MESSAGE = "Please try again or contact support@cleanlab.ai if the issue persists."

    def __init__(self, error_type: str, error_message: str) -> None:
        self.error_type = error_type
        self.error_message = error_message

    def __str__(self) -> str:
        error_msg = f"{self.error_type}\n"
        if self.error_message:
            error_msg += f"{self.error_message}\n"
        error_msg += f"{self.DEFAULT_ERROR_MESSAGE}"
        return error_msg


class InvalidSchemaTypeError(ValueError):
    def __init__(self, msg: str) -> None:
        self.msg = msg

    def __str__(self) -> str:
        return f"{self.msg}\nSee [/guide/concepts/datasets/#schemas] for more information."


class InvalidProjectConfiguration(HandledError):
    pass


class DeploymentError(HandledError):
    pass


class InvalidFilepathError(HandledError):
    def __init__(self, filepath: Union[str, pathlib.Path] = "") -> None:
        if isinstance(filepath, pathlib.Path):
            filepath = str(filepath)
        super().__init__(f"File could not be found at {filepath}. Please check the file path.")
