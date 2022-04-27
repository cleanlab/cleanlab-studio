import json
from cleanlab_cli.decorators import auth_config, previous_state
from cleanlab_cli import api_service
from cleanlab_cli.dataset.util import get_dataset_columns, get_num_rows
from cleanlab_cli.dataset.upload_helpers import upload_rows
from cleanlab_cli.dataset.schema_helpers import (
    load_schema,
    validate_schema,
    propose_schema,
    confirm_schema_save_location,
    save_schema,
)
from cleanlab_cli.click_helpers import *


def resume_upload(api_key, dataset_id, filepath):
    complete = api_service.get_completion_status(api_key, dataset_id)
    if complete:
        abort("Dataset is already fully uploaded.")
    saved_schema = api_service.get_dataset_schema(api_key, dataset_id)
    existing_ids = api_service.get_existing_ids(api_key, dataset_id)
    upload_rows(api_key, dataset_id, filepath, saved_schema, existing_ids)
    return


def upload_with_schema(api_key, schema, columns, filepath):
    progress("Validating provided schema...")
    loaded_schema = load_schema(schema)
    try:
        validate_schema(loaded_schema, columns)
    except ValueError as e:
        abort(str(e))
    success("Provided schema is valid!")
    progress("Initializing dataset...")
    dataset_id = api_service.initialize_dataset(api_key, loaded_schema)
    info(f"Dataset initialized with ID: {dataset_id}")
    upload_rows(api_key=api_key, dataset_id=dataset_id, filepath=filepath, schema=loaded_schema)
    return


@click.command(help="upload your dataset to Cleanlab Studio")
@click.option(
    "--filepath",
    "-f",
    type=click.Path(),
    help="Dataset filepath",
    required=True,
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
        if prev_state.command != "dataset upload":
            abort("Previous command was not a dataset upload, so there is nothing to resume.")

    curr_command_dict = dict(
        command="dataset upload",
        filepath=filepath,
        id=id,
        schema=schema,
        id_column=id_column,
        name=name,
        output=output,
    )
    if prev_state.same_command(curr_command_dict) and not resume:
        prev_dataset_id = prev_state.dataset_id
        if prev_dataset_id is None:
            if prev_state.complete:
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
    else:
        prev_state.new_state(dict(command=curr_command_dict))

    if resume:
        filepath = prev_state.filepath
        dataset_id = prev_state.dataset_id
        if dataset_id is None:
            abort(
                "There was no dataset initialized with the previous upload command, so there is"
                " nothing to resume. Run this command without the --resume flag."
            )
    else:
        dataset_id = id
        # filetype = get_file_extension(filepath)
    columns = get_dataset_columns(filepath)

    # If resuming upload
    if dataset_id is not None:
        resume_upload(api_key, dataset_id, filepath)

    # This is the first upload
    ## Check if uploading with schema
    if schema is not None:
        upload_with_schema(api_key, schema, columns, filepath)

    ## No schema, propose and confirm a schema
    ### Check that all required arguments are present
    if modality is None:
        abort("You must specify a modality (--modality <MODALITY>) for a new dataset upload.")

    if id_column is None:
        abort(
            "You must specify an ID column (--id-column <ID column name>) for a new dataset upload."
        )

    if id_column not in columns:
        abort(f"Could not find specified ID column '{id_column}' in dataset columns.")

    num_rows = get_num_rows(filepath)

    ### Propose schema
    proposed_schema = propose_schema(filepath, columns, id_column, modality, name, num_rows)
    log(json.dumps(proposed_schema, indent=2))
    info(f"No schema was provided. We propose the above schema based on your dataset.")

    proceed_upload = click.confirm("\nUse this schema?", default=None)
    if not proceed_upload:
        info(
            "Proposed schema rejected. Please submit your own schema using --schema.\n",
        )

    if output:
        save_schema(proposed_schema, output)
    else:
        save_filename = confirm_schema_save_location()
        save_schema(proposed_schema, save_filename)

    if proceed_upload:
        dataset_id = api_service.initialize_dataset(api_key, proposed_schema)
        info(f"Dataset initialized with ID: {dataset_id}")
        prev_state.update_state(dict(dataset_id=dataset_id, complete=False))
        upload_rows(
            api_key=api_key, dataset_id=dataset_id, filepath=filepath, schema=proposed_schema
        )
