import json
import pathlib
from typing import Optional
import os

from cleanlab_studio.internal.upload_helpers import (
    get_ingestion_result,
    get_proposed_schema,
    upload_dataset_file,
)

import click

from cleanlab_studio.cli import click_helpers
from cleanlab_studio.cli.click_helpers import abort, info, log
from cleanlab_studio.cli.dataset.schema_helpers import (
    load_schema,
    save_schema,
)
from cleanlab_studio.cli.decorators import auth_config
from cleanlab_studio.cli.decorators.auth_config import AuthConfig
from cleanlab_studio.internal import api
from cleanlab_studio.internal.dataset_source import FilepathDatasetSource
from cleanlab_studio.internal.types import JSONDict
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
    if not os.path.exists(filepath):
        abort(f"cannot upload '{filepath}': no such file or directory")

    dataset_source = FilepathDatasetSource(filepath=pathlib.Path(filepath))
    upload_id = upload_dataset_file(api_key, dataset_source)

    schema: Optional[JSONDict]

    # Check if uploading with schema
    if schema_path is not None:
        schema = load_schema(schema_path)
        proceed_upload = True

    ### Propose schema
    else:
        schema = get_proposed_schema(api_key, upload_id)
        proceed_upload = None
        if schema is None or schema.get("immutable", False):
            proceed_upload = True
        else:
            log(json.dumps(schema, indent=2))
            info(f"No schema was provided. We propose the above schema based on your dataset.")

            proceed_upload = click.confirm("\nUse this schema?", default=True)
            if not proceed_upload:
                info(
                    "Proposed schema rejected. Please submit your own schema using --schema.\n",
                )

            save_filepath = click_helpers.confirm_save_prompt_filepath(
                save_message="Save the generated schema?",
                save_default=None,
                prompt_message="Specify a filename for the schema. Leave this blank to use default",
                prompt_default="schema.json",
                no_save_message="Schema was not saved.",
            )
            if save_filepath:
                save_schema(schema, save_filepath)

    if proceed_upload or schema is None:
        api.confirm_schema(api_key, schema, upload_id)
        dataset_id = get_ingestion_result(api_key, upload_id)
        log(f"Successfully uploaded dataset with ID: {dataset_id}")
        click_helpers.success(
            "Upload completed. View your uploaded dataset at https://app.cleanlab.ai"
        )
