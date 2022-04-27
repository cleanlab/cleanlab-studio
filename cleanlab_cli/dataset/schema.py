import click
from cleanlab_cli.dataset.schema_helpers import (
    load_schema,
    validate_schema,
    propose_schema,
    confirm_schema_save_location,
    save_schema,
    _find_best_matching_column,
)
from cleanlab_cli.dataset.util import get_dataset_columns, get_num_rows
from cleanlab_cli.decorators import previous_state
import json
from cleanlab_cli.click_helpers import (
    abort,
    success,
    info,
    prompt_for_filepath,
    prompt_with_optional_default,
)


@click.group(help="generate and validate dataset schema, or check your dataset against a schema")
def schema():
    pass


@schema.command(name="validate", help="validate an existing schema")
@click.option("--schema", "-s", type=click.Path(), help="Schema filepath")
@click.option("--dataset", "-d", type=click.Path(), help="Dataset filepath", required=False)
@previous_state
def validate_schema_command(prev_state, schema, dataset):
    if schema is None:
        schema = prompt_for_filepath("Specify your schema filepath")
    prev_state.new_state(
        dict(command=dict(command="validate_schema", schema=schema, dataset=dataset))
    )
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


@schema.command(name="generate", help="generate a schema based on your dataset")
@click.option("--filepath", "-f", type=click.Path(), help="Dataset filepath")
@click.option("--output", "-o", type=click.Path(), help="Output filepath")
@click.option(
    "--id-column",
    type=str,
    prompt=True,
    help="Name of ID column in the dataset",
)
@click.option(
    "--modality",
    "--m",
    prompt=True,
    type=click.Choice(["text", "tabular"]),
    help="Dataset modality: text or tabular",
)
@click.option(
    "--name",
    type=str,
    help="Custom name for dataset",
)
@previous_state
def generate_schema_command(prev_state, filepath, output, id_column, modality, name):
    if filepath is None:
        filepath = prompt_for_filepath("Specify your dataset filepath")

    columns = get_dataset_columns(filepath)
    id_column_guess = _find_best_matching_column("id", columns)
    while id_column not in columns:
        id_column = prompt_with_optional_default(
            "Specify the name of the ID column in your dataset.", default=id_column_guess
        )

    prev_state.new_state(
        dict(
            command=dict(
                command="generate schema",
                filepath=filepath,
                output=output,
                id_column=id_column,
                modality=modality,
                name=name,
            ),
        )
    )
    num_rows = get_num_rows(filepath)
    cols = get_dataset_columns(filepath)
    proposed_schema = propose_schema(filepath, cols, id_column, modality, name, num_rows)
    click.echo(json.dumps(proposed_schema, indent=2))
    if output is None:
        output = confirm_schema_save_location()
        save_schema(proposed_schema, output)
