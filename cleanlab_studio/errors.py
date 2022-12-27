class InvalidDatasetError(ValueError):
    pass


class EmptyDatasetError(InvalidDatasetError):
    pass


class ColumnMismatchError(InvalidDatasetError):
    pass
