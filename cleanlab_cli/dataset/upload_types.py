from enum import Enum
from typing import Dict, List, Collection
from dataclasses import dataclass


class ValidationWarning(Enum):
    MISSING_ID = "MISSING_ID"
    MISSING_VAL = "MISSING_VAL"
    MISSING_FILE = "MISSING_FILE"
    UNREADABLE_FILE = "UNREADABLE_FILE"
    TYPE_MISMATCH = "TYPE_MISMATCH"
    DUPLICATE_ID = "DUPLICATE_ID"


def warning_to_readable_name(warning: ValidationWarning) -> str:
    return {
        ValidationWarning.MISSING_ID: "Rows with missing IDs (rows are dropped)",
        ValidationWarning.MISSING_FILE: "Rows with non-existent filepaths (rows are dropped)",
        ValidationWarning.MISSING_VAL: "Rows with missing values (values replaced with null)",
        ValidationWarning.UNREADABLE_FILE: "Rows with unreadable files (rows are dropped)",
        ValidationWarning.TYPE_MISMATCH: (
            "Rows with values that do not match the schema (values replaced with null)"
        ),
        ValidationWarning.DUPLICATE_ID: (
            "Rows with duplicate IDs (only the first row instance is kept, all later rows dropped)"
        ),
    }[warning]


RowWarningsType = Dict[ValidationWarning, List[str]]


@dataclass
class WarningLog:
    MISSING_ID: List[str]
    MISSING_VAL: Dict[str, List[str]]
    MISSING_FILE: Dict[str, List[str]]
    UNREADABLE_FILE: Dict[str, List[str]]
    TYPE_MISMATCH: Dict[str, List[str]]
    DUPLICATE_ID: Dict[str, List[str]]

    @staticmethod
    def init() -> "WarningLog":
        return WarningLog(
            MISSING_ID=list(),
            MISSING_VAL=dict(),
            MISSING_FILE=dict(),
            UNREADABLE_FILE=dict(),
            TYPE_MISMATCH=dict(),
            DUPLICATE_ID=dict(),
        )

    def get(self, key: ValidationWarning) -> Collection[str]:
        return {
            ValidationWarning.MISSING_ID: self.MISSING_ID,
            ValidationWarning.MISSING_VAL: self.MISSING_VAL,
            ValidationWarning.MISSING_FILE: self.MISSING_FILE,
            ValidationWarning.UNREADABLE_FILE: self.UNREADABLE_FILE,
            ValidationWarning.TYPE_MISMATCH: self.TYPE_MISMATCH,
            ValidationWarning.DUPLICATE_ID: self.DUPLICATE_ID,
        }[key]

    def to_dict(self, readable: bool = False) -> Dict[str, Collection[str]]:
        retval = dict()
        for warning_type in ValidationWarning:
            key = warning_to_readable_name(warning_type) if readable else warning_type.value
            retval[key] = self.get(warning_type)
        return retval
