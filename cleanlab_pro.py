import click
from click import ClickException, style
from util import *

class AuthConfig:
    def __init__(self):
        self.username = None
        self.password = None
        self.bearer = None

    def status(self):
        if self.bearer:
            click.echo("Currently logged in!")
        else:
            click.echo("Not logged in.")


auth_config = click.make_pass_decorator(AuthConfig, ensure=True)


@click.group()
@auth_config
def main(config):
    pass


@main.command()
@auth_config
def login(config):
    click.echo("I am logged in!")


@main.group()
@auth_config
def dataset(config):
    pass

# @dataset.command()
# @click.option('-f', '--file', type=click.Path(), help='Dataset filepath', required=True)
# def generate_schema():
#     pass

@dataset.command()
@click.option('--filepath', '-f', type=click.Path(), prompt=True, help='Dataset filepath', required=True)
@click.option('--modality', '-m', type=str, prompt=True, help="Data modality: text, tabular, or image")
@click.option('--id_col', type=str, prompt=True, help="Name of ID column; optional for JSON datasets")
@click.option('--name', type=str, help='Name of dataset')
@click.option('--id', type=str, help="If resuming upload or appending to an existing dataset, specify the dataset ID")
@click.option('--schema', type=click.Path(), help="Filepath to schema JSON file.")
@click.option('--threshold', type=float, default=0.2,
              help="Float between 0 and 1 representing the percentage of null values "
                   "a column is allowed to have, otherwise it is dropped. Default: 0.2")
@auth_config
def upload(config, filepath, modality, id_col, name, id, schema, threshold):
    # Authenticate
    click.echo(config.status())
    filetype = get_file_extension(filepath)

    ## Pre-checks
    if id is None and modality is None:
        raise click.ClickException('You must specify a modality (--modality <MODALITY>) for a new dataset upload.')

    if filetype != 'json' and id_col is None:
        raise click.ClickException('An ID column (--id_col <ID column name>) must be specified for non-JSON datasets.')

    if name is None:
        name = get_filename(filepath)
        click.echo(f"No dataset name provided, setting default filename: {name}\n")
    click.echo(f"Uploading {filepath} with {modality} modality named {name} of ID {id} with schema {schema}\n")

    ## Validation and pre-processing checks

    ### check that ID column exists
    dataset_cols = get_dataset_columns(filepath)
    if filetype != 'json':
        if id_col not in dataset_cols:
            raise ClickException(f"Could not find specified ID column '{id_col}' in dataset columns: {dataset_cols}")

    ### Drop null columns
    null_columns, null_rows, num_rows = diagnose_dataset(filepath, id_col, threshold)

    if len(null_columns) > 0:
        click.secho(
            "We found columns with null values in >= {:.2f}% (--threshold) of rows.".format(threshold * 100),
            fg='red'
        )
        for col in null_columns:
            click.echo(col)
        proceed = click.confirm("Proceed with dropping columns before upload?")
        if not proceed:
            raise ClickException("Columns with null values were not dropped.")

    # Propose and confirm schema
    if schema:
        pass
    else: # generate schema
        proposed_schema = generate_schema(filepath, num_rows)





