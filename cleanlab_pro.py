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


@dataset.command()
@click.option('--filepath', '-f', type=click.Path(), help='Dataset filepath', required=True)
@click.option('--output', '-o', type=click.Path(), help='Output filepath', default='schema.json')
@auth_config
def schema(config, filepath, output):
    null_columns, num_rows = diagnose_dataset(filepath)
    if len(null_columns) > 0:
        click.secho(
            f"Columns with null values in >=20% of rows:",
            fg='red'
        )
        click.echo(f"{null_columns}")
        click.secho("No schema will be generated for these columns.\n", fg='yellow')

    cols = set(get_dataset_columns(filepath)) - set(null_columns)
    retval = propose_schema(filepath, cols, num_rows)
    click.echo(f"Writing schema to {output}\n")
    with open(output, 'w') as f:
        f.write(json.dumps(retval, indent=2))

    click.echo(json.dumps(retval, indent=2))

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
    null_columns, num_rows = diagnose_dataset(filepath, threshold)

    if len(null_columns) > 0:
        click.secho(
            "We found columns with null values in >= {:.2f}% (--threshold) of rows.".format(threshold * 100),
            fg='red'
        )
        for col in null_columns:
            click.echo(col)
        proceed = click.confirm("Proceed with dropping columns before upload?")
        if not proceed:
            raise ClickException(style("Columns with null values were not dropped.", fg='red'))

    kept_columns = set(dataset_cols) - set(null_columns)
    # Propose and confirm schema
    if schema:
        pass
    else: # generate schema
        proposed_schema = propose_schema(filepath, kept_columns, num_rows)
        click.secho(
            f"No schema was provided. We propose the following schema based on your dataset: {proposed_schema}",
            fg='yellow'
        )
        proceed = click.confirm("Use this schema?")
        if not proceed:
            raise ClickException("Proposed schema does not fit use case, please submit your own schema")





