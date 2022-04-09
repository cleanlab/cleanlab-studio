import click
from click import ClickException, style
from cleanlab_cli.dataset.schema_helpers import (
    load_schema,
    validate_schema,
    dump_schema,
    propose_schema,
)
from cleanlab_cli.dataset.util import get_dataset_columns, get_num_rows
import json


@click.group()
def schema():
    pass


@schema.command(name="validate")
@click.option("--filepath", "-f", type=click.Path(), help="Dataset filepath", required=True)
@click.option("--schema", "-s", type=click.Path(), help="Schema filepath", required=True)
def validate_schema_command(filepath, schema):
    cols = get_dataset_columns(filepath)
    loaded_schema = load_schema(schema)
    try:
        validate_schema(loaded_schema, cols)
    except ValueError as e:
        raise ClickException(style(str(e), fg="red"))
    click.secho("Provided schema is valid!", fg="green")


@schema.command(name="generate")
@click.option("--filepath", "-f", type=click.Path(), help="Dataset filepath", required=True)
@click.option("--output", "-o", type=click.Path(), help="Output filepath", default="schema.json")
@click.option(
    "--id_column",
    type=str,
    prompt=True,
    help="If uploading a new dataset without a schema, specify the ID column.",
)
@click.option(
    "--modality",
    "-m",
    prompt=True,
    type=str,
    help=(
        "If uploading a new dataset without a schema, specify data modality: text, tabular, or"
        " image"
    ),
)
@click.option(
    "--name",
    type=str,
    help="If uploading a new dataset without a schema, specify a dataset name.",
)
def generate_schema_command(filepath, output, id_column, modality, name):
    num_rows = get_num_rows(filepath)
    cols = get_dataset_columns(filepath)
    schema = propose_schema(filepath, cols, id_column, modality, name, num_rows)
    click.secho(f"Writing schema to {output}\n")
    dump_schema(output, schema)
    click.echo(json.dumps(schema, indent=2))
    click.secho("\nSaved.", fg="green")
