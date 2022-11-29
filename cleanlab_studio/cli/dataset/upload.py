import csv
import json
from typing import Optional, List
import os
import tempfile

import click

from cleanlab_studio.cli import api_service
from cleanlab_studio.cli import click_helpers
from cleanlab_studio.cli.click_helpers import progress, success, abort, info, log
from cleanlab_studio.cli.dataset.helpers import (
    get_id_column_if_undefined,
    get_filepath_column_if_undefined,
)
from cleanlab_studio.cli.dataset.schema_helpers import (
    load_schema,
    validate_schema,
    propose_schema,
    save_schema,
)
from cleanlab_studio.cli.dataset.upload_helpers import upload_dataset
from cleanlab_studio.cli.dataset.schema_types import Schema
from cleanlab_studio.cli.decorators import auth_config, previous_state
from cleanlab_studio.cli.decorators.auth_config import AuthConfig
from cleanlab_studio.cli.decorators.previous_state import PreviousState
from cleanlab_studio.cli.types import Modality, CommandState, MODALITIES
from cleanlab_studio.cli.util import init_dataset_from_filepath
from cleanlab_studio.version import SCHEMA_VERSION


def resume_upload(api_key: str, dataset_id: str, filepath: str) -> None:
    complete = api_service.get_completion_status(api_key, dataset_id)
    if complete:
        abort("Dataset is already fully uploaded.")
    saved_schema = api_service.get_dataset_schema(api_key, dataset_id)
    existing_ids = {str(x) for x in api_service.get_existing_ids(api_key, dataset_id)}
    upload_dataset(api_key, dataset_id, filepath, saved_schema, existing_ids)


def upload_with_schema(
    api_key: str, schema: str, columns: List[str], filepath: str, prev_state: PreviousState
) -> None:
    progress("Validating provided schema...")
    loaded_schema = load_schema(schema)
    try:
        validate_schema(loaded_schema, columns)
    except ValueError as e:
        abort(str(e))
    success("Provided schema is valid!")
    progress("Initializing dataset...")
    dataset_id = api_service.initialize_dataset(api_key, loaded_schema)
    prev_state.update_args(dict(dataset_id=dataset_id))
    info(f"Dataset initialized with ID: {dataset_id}")
    info(
        "If this upload is interrupted, you may resume it using: cleanlab dataset upload -f"
        f" {filepath} --id {dataset_id}"
    )
    upload_dataset(api_key=api_key, dataset_id=dataset_id, filepath=filepath, schema=loaded_schema)


def simple_image_upload(
    api_key: str, directory: str, dataset_id: Optional[str], prev_state: PreviousState
) -> None:
    classes = sorted(
        [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    )
    metadata = [["id", "path", "label"]]
    for c in classes:
        for f in os.listdir(os.path.join(directory, c)):
            full_path = os.path.join(directory, c, f)
            if not os.path.isfile(full_path) or f.startswith("."):
                # TODO use better heuristics here
                continue
            # using absolute paths here because metadata file will be written to temp dir,
            # and paths are interpreted as relative to that temp dir
            id = os.path.join(c, f)
            metadata.append([id, os.path.abspath(full_path), c])
    # the simplest way to do this is to write metadata to a file and then reuse existing code
    metadata_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv")
    w = csv.writer(metadata_file)
    w.writerows(metadata)
    metadata_file.flush()
    if dataset_id is not None:
        resume_upload(api_key, dataset_id, metadata_file.name)
    else:
        ok = click.confirm(
            f"Upload image dataset with {len(classes)} classes and {len(metadata)-1} images?"
        )
        if not ok:
            abort("aborted")
        constructed_schema = Schema.create(
            metadata={
                "id_column": "id",
                "modality": "image",
                "name": os.path.basename(os.path.abspath(directory)),
                "filepath_column": "path",
            },
            fields={
                "id": {"data_type": "string", "feature_type": "identifier"},
                "path": {"data_type": "string", "feature_type": "filepath"},
                "label": {"data_type": "string", "feature_type": "categorical"},
            },
            version=SCHEMA_VERSION,
        )
        # TODO deduplicate logic with upload_with_schema
        progress("Initializing dataset...")
        dataset_id = api_service.initialize_dataset(api_key, constructed_schema)
        prev_state.update_args(dict(dataset_id=dataset_id))
        info(f"Dataset initialized with ID: {dataset_id}")
        info(
            "If this upload is interrupted, you may resume it using: cleanlab dataset upload -f"
            f" {directory} --id {dataset_id}"
        )
        upload_dataset(
            api_key=api_key,
            dataset_id=dataset_id,
            filepath=metadata_file.name,
            schema=constructed_schema,
        )


@click.command(help="upload your dataset to Cleanlab Studio")
@click.option(
    "--filepath",
    "-f",
    type=click.Path(),
    help="Dataset filepath",
)
@click.option(
    "--id",
    type=str,
    help="If resuming upload or appending to an existing dataset, specify the dataset ID",
)
@click.option(
    "--schema",
    "-s",
    type=click.Path(),
    help="If uploading with a schema, specify the JSON schema filepath.",
)
@click.option(
    "--id-column",
    type=str,
    help="If uploading a new dataset without a schema, specify the ID column.",
)
@click.option(
    "--modality",
    "-m",
    type=click.Choice(MODALITIES),
    help=f"If uploading a new dataset without a schema, specify data modality: {', '.join(MODALITIES)}",
)
@click.option(
    "--filepath-column",
    type=str,
    help=f"If uploading an image dataset, specify the column containing the image filepaths.",
)
@click.option(
    "--name",
    "-n",
    type=str,
    help="If uploading a new dataset without a schema, specify a dataset name.",
)
@click.option(
    "--output", "-o", type=click.Path(), help="Output filepath for issues encountered during upload"
)
@click.option(
    "--resume", "-r", is_flag=True, help="Resume the previous upload if it did not finish"
)
@previous_state
@auth_config
def upload(
    config: AuthConfig,
    prev_state: PreviousState,
    filepath: Optional[str],
    id: Optional[str],
    schema: Optional[str],
    id_column: Optional[str],
    modality: Optional[str],
    filepath_column: Optional[str],
    name: Optional[str],
    output: Optional[str],
    resume: Optional[bool],
) -> None:
    api_key = config.get_api_key()

    if resume:
        if prev_state.get("command") != "dataset upload":
            abort("Previous command was not a dataset upload, so there is nothing to resume.")

    args = dict(
        filepath=filepath,
        dataset_id=id,
        schema=schema,
        id_column=id_column,
        name=name,
        output=output,
    )
    # if dataset upload and not resuming (i.e. not using -r and no --id provided)
    if prev_state.same_command("dataset upload", args) and not resume and not id:
        prev_dataset_id = prev_state.get_arg("dataset_id")
        if prev_dataset_id:  # having a dataset id means it was initialized
            completed = api_service.get_completion_status(api_key, prev_dataset_id)
            if completed:
                proceed = click.confirm(
                    "You previously uploaded a dataset from this filepath with the same"
                    " command arguments. Running this command will generate a new dataset. Do you"
                    " wish to proceed?",
                    default=None,
                )
            else:
                proceed = click.confirm(
                    "You previously partially uploaded a dataset from this filepath with the"
                    " same command arguments. To resume your upload, run 'cleanlab dataset upload"
                    " --resume'. Otherwise, running this command will generate a new dataset. Do"
                    " you wish to proceed?",
                    default=None,
                )
            if not proceed:
                info("Exiting.")
                return

    if resume:
        filepath = prev_state.get_arg("filepath")
        dataset_id = prev_state.get_arg("dataset_id")
        if dataset_id is None:
            abort(
                "There was no dataset initialized with the previous upload command, so there is"
                " nothing to resume. Run this command without the --resume flag."
            )
        args.update(dict(filepath=filepath, dataset_id=dataset_id))
    else:
        dataset_id = id

    command_state: CommandState = dict(command="dataset upload", args=args)
    prev_state.init_state(command_state)

    if filepath is None:
        filepath = click_helpers.prompt_for_filepath("Specify your dataset filepath")
    if not os.path.exists(filepath):
        abort(f"cannot upload '{filepath}': no such file or directory")

    # image modality, simple upload
    if os.path.isdir(filepath) and (modality is None or modality == Modality.image.value):
        if schema is not None:
            abort("-s/--schema is not supported in simple upload mode")
        prev_state.update_args(dict(filepath=filepath))
        simple_image_upload(api_key, filepath, dataset_id, prev_state)
        return

    prev_state.update_args(dict(filepath=filepath))
    dataset = init_dataset_from_filepath(filepath)
    columns = dataset.get_columns()

    # If resuming upload
    if dataset_id is not None:
        resume_upload(api_key, dataset_id, filepath)
        return

    # This is the first upload
    ## Check if uploading with schema
    if schema is not None:
        upload_with_schema(api_key, schema, columns, filepath, prev_state)
        return

    ## No schema, propose and confirm a schema
    ### Check that all required arguments are present

    if modality is None:
        while modality not in MODALITIES:
            modality = click.prompt(f"Specify your dataset modality ({', '.join(MODALITIES)})")

    id_column = get_id_column_if_undefined(id_column=id_column, columns=columns)
    if modality == Modality.image.value:
        filepath_column = get_filepath_column_if_undefined(
            modality=modality, filepath_column=filepath_column, columns=columns
        )

    prev_state.update_args(
        dict(modality=modality, id_column=id_column, filepath_column=filepath_column)
    )

    ### Propose schema
    proposed_schema = propose_schema(
        filepath=filepath,
        columns=columns,
        id_column=id_column,
        modality=modality,
        filepath_column=filepath_column,
        name=name,
    )
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

    if proceed_upload:
        dataset_id = api_service.initialize_dataset(api_key, proposed_schema)
        info(f"Dataset initialized with ID: {dataset_id}")
        prev_state.update_args(dict(dataset_id=dataset_id))
        upload_dataset(
            api_key=api_key, dataset_id=dataset_id, filepath=filepath, schema=proposed_schema
        )
