from typing import Optional, Set, List, Sized, Iterable, cast

import click
from cleanlab_cli.dataset.schema_helpers import (
    load_schema,
    validate_schema,
    propose_schema,
    save_schema,
    _find_best_matching_column,
)
from cleanlab_cli.dataset import upload_helpers
from cleanlab_cli.decorators.previous_state import PreviousState
from cleanlab_cli.dataset.upload_types import ValidationWarning
from cleanlab_cli.types import CommandState, MODALITIES
from cleanlab_cli.util import init_dataset_from_filepath
from cleanlab_cli.decorators import previous_state
import json
from cleanlab_cli.click_helpers import abort, info, success
from cleanlab_cli import click_helpers
from tqdm import tqdm


@click.group(help="generate and validate dataset schema, or check your dataset against a schema")
def schema() -> None:
    pass


@schema.command(name="validate", help="validate an existing schema")
@click.option("--schema", "-s", type=click.Path(), help="Schema filepath")
@click.option("--filepath", "-f", type=click.Path(), help="Dataset filepath")
@previous_state
def validate_schema_command(
    prev_state: PreviousState, schema: Optional[str], filepath: Optional[str]
) -> None:
    if schema is None:
        schema = click_helpers.prompt_for_filepath("Specify your schema filepath")
    command_state: CommandState = dict(
        command="validate schema", args=dict(schema=schema, filepath=filepath)
    )
    prev_state.init_state(command_state)
    loaded_schema = load_schema(schema)
    if filepath:
        dataset = init_dataset_from_filepath(filepath)
        cols = dataset.get_columns()
    else:
        cols = list(loaded_schema.fields)
    try:
        validate_schema(loaded_schema, cols)
    except ValueError as e:
        abort(str(e))
    success("Provided schema is valid!")


@schema.command(name="check", help="check your dataset for type issues based on your schema")
@click.option("--filepath", "-f", type=click.Path(), help="Dataset filepath")
@click.option("--schema", "-s", type=click.Path(), help="Schema filepath")
@click.option("--output", "-o", type=click.Path(), help="Output filepath for type issues found")
@previous_state
def check_dataset_command(
    prev_state: PreviousState, filepath: Optional[str], schema: Optional[str], output: Optional[str]
) -> None:
    if filepath is None:
        filepath = click_helpers.prompt_for_filepath("Specify your dataset filepath")

    if schema is None:
        schema = click_helpers.prompt_for_filepath("Specify your schema filepath")

    command_state: CommandState = dict(
        command="check dataset", args=dict(schema=schema, filepath=filepath)
    )
    prev_state.init_state(command_state)

    dataset = init_dataset_from_filepath(filepath)
    loaded_schema = load_schema(schema)
    log = upload_helpers.create_warning_log()
    num_records = len(dataset)
    seen_ids: Set[str] = set()
    existing_ids: Set[str] = set()

    for record in tqdm(
        dataset.read_streaming_records(), total=num_records, initial=1, leave=True, unit=" rows"
    ):
        row, row_id, warnings = upload_helpers.validate_and_process_record(
            record, loaded_schema, seen_ids, existing_ids
        )
        upload_helpers.update_log_with_warnings(log, row_id, warnings)
        # row and row ID both present, i.e. row will be uploaded
        if row_id:
            seen_ids.add(row_id)

    total_warnings = sum([len(log.get(warning_type)) for warning_type in ValidationWarning])
    if total_warnings == 0:
        success("\nNo type issues were found when checking your dataset. Nice!")
    else:
        info(f"\nFound {total_warnings} type issues when checking your dataset.")
        upload_helpers.echo_log_warnings(log)

        if not output:
            output = click_helpers.confirm_save_prompt_filepath(
                save_message="Would you like to save the issues for viewing?",
                save_default=None,
                prompt_message=(
                    "Specify a filename for the dataset issues. Leave this blank to use default"
                ),
                prompt_default="issues.json",
                no_save_message="Dataset type issues were not saved.",
            )
        if output:
            upload_helpers.save_warning_log(log, output)
            click_helpers.confirm_open_file(
                "Would you like to open your issues file for viewing?", filepath=output
            )

    click.secho("Check completed.", fg="green")


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
    type=click.Choice(MODALITIES),
    help="Dataset modality: text or tabular",
)
@click.option(
    "--name",
    type=str,
    help="Custom name for dataset",
)
@previous_state
def generate_schema_command(
    prev_state: PreviousState,
    filepath: Optional[str],
    output: Optional[str],
    id_column: Optional[str],
    modality: Optional[str],
    name: Optional[str],
) -> None:
    if filepath is None:
        filepath = click_helpers.prompt_for_filepath("Specify your dataset filepath")

    dataset = init_dataset_from_filepath(filepath)
    columns = dataset.get_columns()
    id_column_guess = _find_best_matching_column("id", columns)
    while id_column not in columns:
        id_column = click.prompt(
            "Specify the name of the ID column in your dataset.", default=id_column_guess
        )

    command_state: CommandState = dict(
        command="generate schema",
        args=dict(
            filepath=filepath,
            output=output,
            id_column=id_column,
            modality=modality,
            name=name,
        ),
    )
    prev_state.init_state(command_state)

    proposed_schema = propose_schema(filepath, columns, id_column, modality, name)
    click.echo(json.dumps(proposed_schema.to_dict(), indent=2))
    if not output:
        output = click_helpers.confirm_save_prompt_filepath(
            save_message="Would you like to save the generated schema?",
            save_default=None,
            prompt_message="Specify a filename for the schema. Leave this blank to use default",
            prompt_default="schema.json",
            no_save_message="Schema was not saved.",
        )
        if output is None:
            return
    save_schema(proposed_schema, output)
    click_helpers.confirm_open_file(
        message="Would you like to open your schema file?", filepath=output
    )
