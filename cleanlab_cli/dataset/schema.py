import click
from cleanlab_cli.dataset.schema_helpers import (
    load_schema,
    validate_schema,
    propose_schema,
    confirm_schema_save_location,
    save_schema,
)
from cleanlab_cli.dataset.util import get_dataset_columns, get_num_rows
import json
from cleanlab_cli.click_helpers import abort, success, info


@click.group()
def schema():
    pass


@schema.command(name="validate")
@click.option("--schema", "-s", type=click.Path(), help="Schema filepath", required=True)
@click.option("--dataset", "-d", type=click.Path(), help="Dataset filepath", required=False)
def validate_schema_command(schema, dataset):
    loaded_schema = load_schema(schema)
    if dataset:
        cols = get_dataset_columns(dataset)
    else:
        cols = list(loaded_schema["fields"])
    try:
        validate_schema(loaded_schema, cols)
    except ValueError as e:
        abort(str(e))
    success("Provided schema is valid!")


@schema.command(name="generate")
@click.option("--filepath", "-f", type=click.Path(), help="Dataset filepath", required=True)
@click.option("--output", "-o", type=click.Path(), help="Output filepath")
@click.option(
    "--id_column",
    type=str,
    prompt=True,
    help="If uploading a new dataset without a schema, specify the ID column.",
)
@click.option(
    "--modality",
    "--m",
    prompt=True,
    type=click.Choice(["text", "tabular"]),
    help="If uploading a new dataset without a schema, specify data modality: text, tabular",
)
@click.option(
    "--name",
    type=str,
    help="If uploading a new dataset without a schema, specify a dataset name.",
)
def generate_schema_command(filepath, output, id_column, modality, name):
    num_rows = get_num_rows(filepath)
    cols = get_dataset_columns(filepath)
    proposed_schema = propose_schema(filepath, cols, id_column, modality, name, num_rows)
    click.echo(json.dumps(proposed_schema, indent=2))
    if output is None:
        output = confirm_schema_save_location()
        save_schema(proposed_schema, output)
