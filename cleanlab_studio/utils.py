from typing import List

from .constants import CL_COLUMN_NAMES


def get_cl_column_names(id_col: str) -> List[str]:
    return [id_col] + CL_COLUMN_NAMES
