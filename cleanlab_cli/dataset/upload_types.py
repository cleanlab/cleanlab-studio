from enum import Enum
from typing import Dict, List, Union, Collection
from dataclasses import dataclass

from cleanlab_cli.dataset import warning_to_readable_name


class ValidationWarning(Enum):
    MISSING_ID = "MISSING_ID"
    MISSING_VAL = "MISSING_VAL"
    TYPE_MISMATCH = "TYPE_MISMATCH"
    DUPLICATE_ID = "DUPLICATE_ID"


RowWarningsType = Dict[ValidationWarning, List[str]]


@dataclass
class WarningLog:
    MISSING_ID: List[str]
    MISSING_VAL: Dict[str, List[str]]
    TYPE_MISMATCH: Dict[str, List[str]]
    DUPLICATE_ID: Dict[str, List[str]]

    @staticmethod
    def init() -> "WarningLog":
        return WarningLog(
            MISSING_ID=list(), MISSING_VAL=dict(), TYPE_MISMATCH=dict(), DUPLICATE_ID=dict()
        )

    def get(self, key: ValidationWarning) -> Collection[str]:
        return {
            ValidationWarning.MISSING_ID: self.MISSING_ID,
            ValidationWarning.MISSING_VAL: self.MISSING_VAL,
            ValidationWarning.TYPE_MISMATCH: self.TYPE_MISMATCH,
            ValidationWarning.DUPLICATE_ID: self.DUPLICATE_ID,
        }[key]

    def to_dict(self, readable: bool = False) -> Dict[str, Collection[str]]:
        retval = dict()
        for warning_type in ValidationWarning:
            key = warning_to_readable_name(warning_type) if readable else warning_type.value
            retval[key] = self.get(warning_type)
        return retval
