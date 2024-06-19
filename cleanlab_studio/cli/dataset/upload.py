import pathlib
from typing import cast, List, Optional

from cleanlab_studio.internal.types import SchemaOverride
from cleanlab_studio.internal.upload_helpers import upload_dataset

import click

from cleanlab_studio.cli import click_helpers
from cleanlab_studio.cli.click_helpers import abort
from cleanlab_studio.cli.dataset.schema_helpers import (
    load_schema_overrides,
)
from cleanlab_studio.cli.decorators import auth_config
from cleanlab_studio.cli.decorators.auth_config import AuthConfig
from cleanlab_studio.internal.dataset_source import FilepathDatasetSource
from cleanlab_studio.internal.util import telemetry


@click.command(help="upload your dataset to Cleanlab Studio")
@click.argument(
    "filepath",
    type=click.Path(),
    required=False,
    default=None,
)
@click.option(
    "--schema_path",
    "-s",
    type=click.Path(),
    help="If uploading with a schema, specify the JSON schema filepath.",
)
@auth_config
@telemetry(load_api_key=True)
def upload(
    config: AuthConfig,
    filepath: Optional[str],
    schema_path: Optional[str],
) -> None:
    api_key = config.get_api_key()

    if filepath is None:
        filepath = click_helpers.prompt_for_filepath("Specify your dataset filepath")

    dataset_source = FilepathDatasetSource(filepath=pathlib.Path(filepath))

    schema_overrides = None
    if schema_path:
        schema_overrides = load_schema_overrides(schema_path)

    dataset_id = upload_dataset(
        api_key,
        dataset_source,
        schema_overrides=schema_overrides,
    )

    click_helpers.success(
        f"Upload completed (dataset ID = {dataset_id}). View your uploaded dataset at https://app.cleanlab.ai"
    )
