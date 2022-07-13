import json

import click

from cleanlab_cli import api_service
from cleanlab_cli import click_helpers
from cleanlab_cli.click_helpers import progress, success, abort, info, log
from cleanlab_cli.dataset.schema_helpers import (
    load_schema,
    validate_schema,
    propose_schema,
    save_schema,
    _find_best_matching_column,
)
from cleanlab_cli.dataset.upload_helpers import upload_dataset
from cleanlab_cli.decorators import auth_config, previous_state
from cleanlab_cli.util import init_dataset_from_filepath


def resume_upload(api_key, dataset_id, filepath):
    complete = api_service.get_completion_status(api_key, dataset_id)
    if complete:
        abort("Dataset is already fully uploaded.")
    saved_schema = api_service.get_dataset_schema(api_key, dataset_id)
    existing_ids = api_service.get_existing_ids(api_key, dataset_id)
    upload_dataset(api_key, dataset_id, filepath, saved_schema, existing_ids)
    return


def upload_with_schema(api_key, schema, columns, filepath, prev_state):
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
    return


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
    type=click.Choice(["text", "tabular"]),
    help="If uploading a new dataset without a schema, specify data modality: text, tabular",
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
def upload(config, prev_state, filepath, id, schema, id_column, modality, name, output, resume):
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

    prev_state.init_state(dict(command="dataset upload", args=args))

    if filepath is None:
        filepath = click_helpers.prompt_for_filepath("Specify your dataset filepath")

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
        while modality not in ["text", "tabular"]:
            modality = click.prompt("Specify your dataset modality (text, tabular)")

    if id_column is None:
        id_column_guess = _find_best_matching_column("id", columns)
        while id_column not in columns:
            id_column = click.prompt(
                "Specify the name of the ID column in your dataset.", default=id_column_guess
            )

    prev_state.update_args(dict(modality=modality, id_column=id_column))

    ### Propose schema
    proposed_schema = propose_schema(filepath, columns, id_column, modality, name)
    log(json.dumps(proposed_schema, indent=2))
    info(f"No schema was provided. We propose the above schema based on your dataset.")

    proceed_upload = click.confirm("\nUse this schema?", default=None)
    if not proceed_upload:
        info(
            "Proposed schema rejected. Please submit your own schema using --schema.\n",
        )

    save_filepath = click_helpers.confirm_save_prompt_filepath(
        save_message="Would you like to save the generated schema?",
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
