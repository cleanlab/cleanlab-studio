import json
import pathlib
from typing import Optional
import os

import click

from cleanlab_studio.cli import api_service
from cleanlab_studio.cli import click_helpers
from cleanlab_studio.cli.click_helpers import progress, success, abort, info, log
from cleanlab_studio.cli.dataset.schema_helpers import (
    load_schema,
    validate_schema,
    save_schema,
)
from cleanlab_studio.cli.dataset.upload_helpers import (
    upload_dataset_file,
    get_proposed_schema,
    get_ingestion_result,
)
from cleanlab_studio.cli.decorators import auth_config
from cleanlab_studio.cli.decorators.auth_config import AuthConfig
from cleanlab_studio.cli.types import MODALITIES


def upload_with_schema(
    api_key: str,
    schema: str,
    upload_id: str,
) -> None:
    progress("Validating provided schema...")
    loaded_schema = load_schema(schema)
    try:
        validate_schema(loaded_schema)
    except ValueError as e:
        abort(str(e))
    success("Provided schema is valid!")
    api_service.confirm_schema(api_key, loaded_schema, upload_id)
    dataset_id = get_ingestion_result(api_key, upload_id)
    log(f"Successfully uploaded dataset with ID: {dataset_id}")
    click_helpers.success("Upload completed. View your uploaded dataset at https://app.cleanla.ai")


@click.command(help="upload your dataset to Cleanlab Studio")
@click.option(
    "--filepath",
    "-f",
    type=click.Path(),
    help="Dataset filepath",
)
@click.option(
    "--schema",
    "-s",
    type=click.Path(),
    help="If uploading with a schema, specify the JSON schema filepath.",
)
@click.option(
    "--modality",
    "-m",
    type=click.Choice(MODALITIES),
    help=f"If uploading a new dataset without a schema, specify data modality: {', '.join(MODALITIES)}",
)
@auth_config
def upload(
    config: AuthConfig,
    filepath: Optional[str],
    schema: Optional[str],
    modality: Optional[str],
) -> None:
    api_key = config.get_api_key()

    if filepath is None:
        filepath = click_helpers.prompt_for_filepath("Specify your dataset filepath")
    if not os.path.exists(filepath):
        abort(f"cannot upload '{filepath}': no such file or directory")

    upload_id = upload_dataset_file(api_key, pathlib.Path(filepath))

    # Check if uploading with schema
    if schema is not None:
        if pathlib.Path(filepath).suffix == ".zip":
            upload_with_schema(api_key, schema, upload_id)
        else:
            upload_with_schema(api_key, schema, upload_id)
        return

    ## No schema, propose and confirm a schema
    ### Check that all required arguments are present
    if modality is None:
        while modality not in MODALITIES:
            modality = click.prompt(f"Specify your dataset modality ({', '.join(MODALITIES)})")

    ### Propose schema
    proposed_schema = get_proposed_schema(api_key, upload_id)
    proceed_upload = None
    if proposed_schema is not None:
        log(json.dumps(proposed_schema.to_dict(), indent=2))
        info(f"No schema was provided. We propose the above schema based on your dataset.")

        proceed_upload = click.confirm("\nUse this schema?", default=None)
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
            save_schema(proposed_schema, save_filepath)

    if proceed_upload or proposed_schema is None:
        api_service.confirm_schema(api_key, proposed_schema, upload_id)
        dataset_id = get_ingestion_result(api_key, upload_id)
        log(f"Successfully uploaded dataset with ID: {dataset_id}")
        click_helpers.success(
            "Upload completed. View your uploaded dataset at https://app.cleanlab.ai"
        )
