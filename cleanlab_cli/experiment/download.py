import os

import click
import pandas as pd

from cleanlab_cli import api_service
from cleanlab_cli import click_helpers
from cleanlab_cli import util
from cleanlab_cli.click_helpers import log
from cleanlab_cli.decorators import previous_state, auth_config
from cleanlab_cli.settings import CleanlabSettings


@click.command(help="export cleaned dataset labels")
@click.option(
    "--id",
    type=str,
    prompt=True,
    help="experiment ID",
)
@click.option(
    "--combine",
    "-c",
    is_flag=True,
    help=(
        "Add clean dataset labels with original dataset as a new column 'Cleanlab_labels'."
        " --filepath must be provided."
    ),
)
@click.option(
    "--filepath",
    "-f",
    type=click.Path(),
    help="Set only if using the --combine flag. Filepath to original dataset.",
)
@click.option(
    "--output",
    "-f",
    type=click.Path(),
    help="Output for cleaned labels or dataset combined with cleaned labels (if --combine is set).",
)
@previous_state
@auth_config
def download(config, prev_state, id, combine, filepath, output):
    prev_state.init_state(dict(command="download labels", args=dict(id=id)))
    CleanlabSettings.init_cleanlab_dir()
    api_key = config.get_api_key()

    rows, id_column = api_service.download_cleaned_labels(api_key, experiment_id=id)

    clean_df = pd.DataFrame(rows)

    if combine:
        if filepath is None:
            filepath = click_helpers.prompt_for_filepath("Specify your dataset filepath")
        else:
            if not os.path.exists(filepath):
                log(f"Specified file {filepath} could not be found.")
                filepath = click_helpers.prompt_for_filepath("Specify your dataset filepath")

        filename = util.get_filename(filepath)
        if output is None:
            output = click.prompt(
                "Specify your output filepath. Leave blank to use default",
                default=f"cleaned_{filename}",
            )

        util.combine_fields_with_dataset(filepath, id_column, output)
    else:
        while output is None or util.get_file_extension(output) != ".csv":
            output = click.prompt(
                "Specify your output filepath (must be .csv). Leave blank to use default",
                default=f"clean_labels.csv",
            )
        clean_df.to_csv(output, index=False)
