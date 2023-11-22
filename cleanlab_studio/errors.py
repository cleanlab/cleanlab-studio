class InvalidDatasetError(ValueError):
    pass


class EmptyDatasetError(InvalidDatasetError):
    pass


class ColumnMismatchError(InvalidDatasetError):
    pass


class APIError(Exception):
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
