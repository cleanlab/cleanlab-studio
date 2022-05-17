import os
from collections import defaultdict

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
    "--filepath",
    "-f",
    type=click.Path(),
    help="Set a filepath to original dataset.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output for cleaned labels or dataset combined with cleaned labels.",
)
@previous_state
@auth_config
def download(config, prev_state, id, filepath, output):
    prev_state.init_state(dict(command="download labels", args=dict(id=id)))
    CleanlabSettings.init_cleanlab_dir()
    api_key = config.get_api_key()

    rows = api_service.download_clean_labels(api_key, experiment_id=id)
    clean_df = pd.DataFrame(rows, columns=["id", "clean_label"])

    if filepath:
        id_column = api_service.get_id_column(api_key, experiment_id=id)
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

        ids_to_fields_to_values = defaultdict(dict)
        for id, clean_label in zip(clean_df["id"], clean_df["clean_label"]):
            ids_to_fields_to_values[id]["clean_label"] = clean_label

        util.combine_fields_with_dataset(filepath, id_column, ids_to_fields_to_values, output)
        click_helpers.success(f"Saved to {output}")
    else:
        while output is None or util.get_file_extension(output) != ".csv":
            output = click.prompt(
                "Specify your output filepath (must be .csv). Leave blank to use default",
                default=f"clean_labels.csv",
            )
        clean_df.to_csv(output, index=False)
        click_helpers.success(f"Saved to {output}")
