from cleanlab_cli.dataset.util import *
import click
from click import ClickException, style
from cleanlab_cli.auth.auth import auth_config


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
def upload(config, filepath, dataset_id, schema, id_column, modality, name):
    # Authenticate
    click.echo(config.status())
    filetype = get_file_extension(filepath)
    columns = get_dataset_columns(filepath)

    # Check if resuming upload
    if dataset_id is not None:
        saved_schema = get_dataset_schema(dataset_id)
        existing_ids = get_existing_ids(dataset_id)
        upload_rows(filepath, saved_schema, existing_ids)
        return

    # First upload
    ## Check if uploading with schema
    if schema is not None:
        click.secho("Validating provided schema...", fg="yellow")
        loaded_schema = load_schema(schema)
        try:
            validate_schema(loaded_schema, columns)
        except ValueError as e:
            raise ClickException(style(str(e), fg="red"))
        click.secho("Provided schema is valid!", fg="green")
        click.secho("Uploading rows...", fg="yellow")
        upload_rows(filepath, loaded_schema)
        return

    ## No schema, propose and confirm a schema
    ### Check that all required arguments are present
    if modality is None:
        raise click.ClickException(
            style(
                "You must specify a modality (--modality <MODALITY>) for a new dataset upload.",
                fg="red",
            )
        )

    if id_column is None:
        raise click.ClickException(
            style(
                "You must specify an ID column (--id_column <ID column name>) for a new dataset"
                " upload.",
                fg="red",
            )
        )

    if id_column not in columns:
        raise ClickException(
            style(
                f"Could not find specified ID column '{id_column}' in dataset columns.",
                fg="red",
            )
        )

    num_rows = get_num_rows(filepath)

    ### Propose schema
    proposed_schema = propose_schema(filepath, columns, id_column, modality, name, num_rows)
    click.secho(
        f"No schema was provided. We propose the following schema based on your dataset:",
        fg="yellow",
    )
    click.echo(json.dumps(proposed_schema, indent=2))

    proceed_upload = click.confirm("\n\nUse this schema?")
    if not proceed_upload:
        click.secho(
            "Proposed schema rejected. Please submit your own schema using --schema. A starter"
            " schema can be generated for your dataset using 'cleanlab dataset schema -f"
            " <filepath>'\n\n",
            fg="red",
        )

    save_schema = click.prompt("Would you like to save the generated schema to 'schema.json'?")
    if save_schema:
        dump_schema("schema.json", proposed_schema)
        click.secho("Saved schema to 'schema.json'.", fg="green")

    if proceed_upload:
        upload_rows(filepath, proposed_schema)
