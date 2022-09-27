from typing import Optional, List, Set, Any

import click
from tqdm import tqdm
import queue

from cleanlab_cli.classes.dataset import Dataset
from cleanlab_cli.dataset.upload_helpers import (
    validate_and_process_record,
    update_log_with_warnings,
)
from cleanlab_cli.dataset.schema_types import Schema
from cleanlab_cli.dataset.upload_types import WarningLog
from cleanlab_cli.dataset.schema_helpers import _find_best_matching_column


def get_id_column_if_undefined(id_column: Optional[str], columns: List[str]) -> str:
    if id_column is None:
        id_column_guess = _find_best_matching_column("id", columns)
        while id_column not in columns:
            id_column = click.prompt(
                "Specify the name of the ID column in your dataset.", default=id_column_guess
            )
    return id_column


def get_filepath_column_if_undefined(
    modality: str, columns: List[str], filepath_column: Optional[str]
) -> str:
    if filepath_column is None:
        filepath_column_guess = _find_best_matching_column("filepath", columns)
        while filepath_column not in columns:
            filepath_column = click.prompt(
                f"Specify the name of the filepath column (containing your {modality} filepaths) in your dataset.",
                default=filepath_column_guess,
            )
    return filepath_column
