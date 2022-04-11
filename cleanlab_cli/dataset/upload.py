import json
from cleanlab_cli.auth.auth import auth_config
from cleanlab_cli import api_service
from cleanlab_cli.dataset.util import get_dataset_columns, get_num_rows
from cleanlab_cli.dataset.upload_helpers import upload_rows
from cleanlab_cli.dataset.schema_helpers import (
    load_schema,
    validate_schema,
    propose_schema,
    dump_schema,
)
from cleanlab_cli.click_helpers import *


@click.command()
@click.option(
    "--filepath",
    "-f",
    type=click.Path(),
    prompt=True,
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
    type=click.Path(),
    help="If uploading with a schema, specify the JSON schema filepath.",
)
@click.option(
    "--id_column",
    type=str,
    help="If uploading a new dataset without a schema, specify the ID column.",
)
@click.option(
    "--modality",
    "-m",
    type=str,
    help="If uploading a new dataset without a schema, specify data modality: text, tabular",
)
@click.option(
    "--name",
    type=str,
    help="If uploading a new dataset without a schema, specify a dataset name.",
)
@auth_config
def upload(config, filepath, id, schema, id_column, modality, name):
    dataset_id = id
    api_key = config.get_api_key()
    # filetype = get_file_extension(filepath)
    columns = get_dataset_columns(filepath)

    # If resuming upload
    if dataset_id is not None:
        complete = api_service.get_completion_status(api_key, dataset_id)
        if complete:
            abort("Dataset has already been uploaded fully.")
        saved_schema = api_service.get_dataset_schema(api_key, dataset_id)
        existing_ids = api_service.get_existing_ids(api_key, dataset_id)
        upload_rows(api_key, dataset_id, filepath, saved_schema, existing_ids)
        return

    # This is the first upload
    ## Check if uploading with schema
    if schema is not None:
        progress("Validating provided schema...")
        loaded_schema = load_schema(schema)
        try:
            validate_schema(loaded_schema, columns)
        except ValueError as e:
            abort(str(e))
        success("Provided schema is valid!")
        progress("Initializing dataset...")
        res = api_service.initialize_dataset(api_key, schema)
        dataset_id = res.json().dataset_id
        info(
            f"Dataset has been initialized with ID: {dataset_id}. Save this dataset ID for future"
            " use."
        )
        progress("Uploading rows...")
        upload_rows(api_key=api_key, dataset_id=dataset_id, filepath=filepath, schema=loaded_schema)

    ## No schema, propose and confirm a schema
    ### Check that all required arguments are present
    if modality is None:
        abort("You must specify a modality (--modality <MODALITY>) for a new dataset upload.")

    if id_column is None:
        abort(
            "You must specify an ID column (--id_column <ID column name>) for a new dataset upload."
        )

    if id_column not in columns:
        abort(f"Could not find specified ID column '{id_column}' in dataset columns.")

    num_rows = get_num_rows(filepath)

    ### Propose schema
    proposed_schema = propose_schema(filepath, columns, id_column, modality, name, num_rows)
    info(f"No schema was provided. We propose the following schema based on your dataset:")
    log(json.dumps(proposed_schema, indent=2))

    proceed_upload = click.confirm("\n\nUse this schema?")
    if not proceed_upload:
        warn(
            "Proposed schema rejected. Please submit your own schema using --schema. A starter"
            " schema can be generated for your dataset using 'cleanlab dataset schema -f"
            " <filepath>'\n\n",
        )

    save_schema = click.confirm(
        "Would you like to save the generated schema to 'schema.json'?",
    )

    if save_schema:
        dump_schema("schema.json", proposed_schema)
        success("Saved schema to 'schema.json'.")

    if proceed_upload:
        dataset_id = api_service.initialize_dataset(api_key, proposed_schema)
        upload_rows(
            api_key=api_key, dataset_id=dataset_id, filepath=filepath, schema=proposed_schema
        )
