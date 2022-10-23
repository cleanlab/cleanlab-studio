import os
from collections import defaultdict
from typing import Dict, Any, Optional

import click
import pandas as pd

from cleanlab_studio.cli import api_service
from cleanlab_studio.cli import click_helpers
from cleanlab_studio.cli import util
from cleanlab_studio.cli.cleanset.download_helpers import combine_fields_with_dataset
from cleanlab_studio.cli.click_helpers import log, progress
from cleanlab_studio.cli.decorators import previous_state, auth_config
from cleanlab_studio.cli.decorators.auth_config import AuthConfig
from cleanlab_studio.cli.decorators.previous_state import PreviousState
from cleanlab_studio.cli.settings import CleanlabSettings
from cleanlab_studio.cli.types import IDType, RecordType, DatasetFileExtension


@click.command(help="download Cleanlab columns")
@click.option(
    "--id",
    type=str,
    prompt=True,
    help="cleanset ID",
)
@click.option(
    "--filepath",
    "-f",
    type=click.Path(),
    help=(
        "To combine Cleanlab columns with existing dataset, set this as the filepath to the"
        " original dataset"
    ),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output filepath",
)
@click.option(
    "--all",
    "-a",
    is_flag=True,
    help=(
        "Set this flag to download all Cleanlab columns (suggested label, clean label, label"
        " quality, issue). Exclude this flag to download only the clean label column."
    ),
)
@previous_state
@auth_config
def download(
    config: AuthConfig,
    prev_state: PreviousState,
    id: str,
    filepath: Optional[str],
    output: Optional[str],
    all: bool,
) -> None:
    prev_state.init_state(dict(command="download labels", args=dict(id=id)))
    CleanlabSettings.init_cleanlab_dir()
    api_key = config.get_api_key()
    progress("Downloading Cleanlab columns...")
    rows = api_service.download_cleanlab_columns(api_key, cleanset_id=id, all=all)

    if all:
        clean_df_columns = ["id", "issue", "label_quality", "suggested_label", "clean_label"]
    else:
        clean_df_columns = ["id", "clean_label"]

    if filepath:
        id_column = api_service.get_id_column(api_key, cleanset_id=id)
        if not os.path.exists(filepath):
            log(f"Specified file {filepath} could not be found.")
            filepath = click_helpers.prompt_for_filepath("Specify your dataset filepath")

        filename = util.get_filename(filepath)
        while output is None:
            output = click.prompt(
                "Specify your output filepath. Leave blank to use default",
                default=f"cleaned_{filename}",
            )
            if os.path.exists(output):
                click_helpers.error(
                    "A file already exists at this filepath, use a different filepath."
                )
                output = None

        clean_df = pd.DataFrame(rows, columns=clean_df_columns).set_index("id")

        ids_to_fields_to_values: Dict[str, RecordType] = defaultdict(dict)
        for row_id, row in clean_df.iterrows():
            fields_to_values = dict(row)
            ids_to_fields_to_values[str(row_id)] = fields_to_values

        combine_fields_with_dataset(filepath, id_column, ids_to_fields_to_values, output)
        click_helpers.success(f"Saved to {output}")
    else:
        while output is None or util.get_dataset_file_extension(output) != DatasetFileExtension.csv:
            output = click.prompt(
                "Specify your output filepath (must be .csv). Leave blank to use default",
                default=f"clean_labels.csv",
            )
        clean_df = pd.DataFrame(rows, columns=clean_df_columns)
        clean_df.to_csv(output, index=False)
        click_helpers.success(f"Saved to {output}")
