from typing import Optional, Set, List, Sized, Iterable, cast

import click

from cleanlab_studio.cli.dataset.helpers import (
    get_id_column_if_undefined,
    get_filepath_column_if_undefined,
)
from cleanlab_studio.cli.dataset.upload_helpers import process_dataset
from cleanlab_studio.cli.dataset.schema_helpers import (
    load_schema,
    validate_schema,
    propose_schema,
    save_schema,
)
from cleanlab_studio.cli.dataset import upload_helpers
from cleanlab_studio.cli.decorators.previous_state import PreviousState
from cleanlab_studio.cli.dataset.upload_types import ValidationWarning
from cleanlab_studio.cli.types import CommandState, MODALITIES, Modality
from cleanlab_studio.cli.util import init_dataset_from_filepath
from cleanlab_studio.cli.decorators import previous_state
import json
from cleanlab_studio.cli.click_helpers import abort, info, success
from cleanlab_studio.cli import click_helpers


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
    seen_ids: Set[str] = set()
    existing_ids: Set[str] = set()

    process_dataset(
        dataset=dataset, schema=loaded_schema, seen_ids=seen_ids, existing_ids=existing_ids, log=log
    )

    total_warnings = sum([len(log.get(warning_type)) for warning_type in ValidationWarning])
    if total_warnings == 0:
        success("\nNo type issues were found when checking your dataset. Nice!")
    else:
        info(f"\nFound {total_warnings} type issues when checking your dataset.")
        upload_helpers.echo_log_warnings(log)

        if not output:
            output = click_helpers.confirm_save_prompt_filepath(
                save_message="Save the issues for viewing?",
                save_default=None,
                prompt_message=(
                    "Specify a filename for the dataset issues. Leave this blank to use default"
                ),
                prompt_default="issues.json",
                no_save_message="Dataset type issues were not saved.",
            )
        if output:
            upload_helpers.save_warning_log(log, output)
            click_helpers.confirm_open_file("Open your issues file for viewing?", filepath=output)

    click_helpers.success("Check completed.")


@schema.command(name="generate", help="generate a schema based on your dataset")
@click.option("--filepath", "-f", type=click.Path(), help="Dataset filepath")
@click.option("--output", "-o", type=click.Path(), help="Output filepath")
@click.option(
    "--id-column",
    type=str,
    help="Name of ID column in the dataset",
)
@click.option(
    "--modality",
    "--m",
    prompt=True,
    type=click.Choice(MODALITIES),
    help=f"Dataset modality: {', '.join(MODALITIES)}",
)
@click.option(
    "--filepath-column",
    type=str,
    help=f"If uploading an image dataset, specify the column containing the image filepaths.",
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
    filepath_column: Optional[str],
    name: Optional[str],
) -> None:
    if filepath is None:
        filepath = click_helpers.prompt_for_filepath("Specify your dataset filepath")

    dataset = init_dataset_from_filepath(filepath)
    columns = dataset.get_columns()
    id_column = get_id_column_if_undefined(id_column=id_column, columns=columns)
    if modality == Modality.image.value:
        filepath_column = get_filepath_column_if_undefined(
            modality=modality, filepath_column=filepath_column, columns=columns
        )

    command_state: CommandState = dict(
        command="generate schema",
        args=dict(
            filepath=filepath,
            output=output,
            id_column=id_column,
            modality=modality,
            filepath_column=filepath_column,
            name=name,
        ),
    )
    prev_state.init_state(command_state)

    proposed_schema = propose_schema(
        filepath=filepath,
        columns=columns,
        id_column=id_column,
        modality=modality,
        filepath_column=filepath_column,
        name=name,
    )
    click.echo(json.dumps(proposed_schema.to_dict(), indent=2))

    if not output:
        output = click_helpers.confirm_save_prompt_filepath(
            save_message="Save the generated schema?",
            save_default=None,
            prompt_message="Specify a filename for the schema. Leave this blank to use default",
            prompt_default="schema.json",
            no_save_message="Schema was not saved.",
        )
        if output is None:
            return

    save_schema(proposed_schema, output)
    click_helpers.confirm_open_file(message="Open your schema file?", filepath=output)
