from cleanlab_cli.dataset.schema_helpers import (
    load_schema,
    validate_schema,
    dump_schema,
    propose_schema,
)
from cleanlab_cli.dataset.util import get_dataset_columns, get_num_rows
import json
from cleanlab_cli.click_helpers import *


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
    click.echo(json.dumps(schema, indent=2))
    if output is None:
        save = click.confirm("Would you like to save the generated schema?")
        if save:
            output = ""
            while not output.endswith(".json"):
                output = click.prompt(
                    "Specify a filename for the schema (e.g. schema.json). Filename must end with"
                    " .json"
                )
            if output == "":
                output = "schema.json"

    if output:
        progress(f"Writing schema to {output}...")
        dump_schema(output, schema)
        success("Saved.")
    else:
        info("Schema was not saved.")
