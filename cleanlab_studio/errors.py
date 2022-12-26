class InvalidDatasetError(RuntimeError):
    ...


class EmptyDatasetError(InvalidDatasetError):
    ...


class ColumnMismatchError(InvalidDatasetError):
    ...
