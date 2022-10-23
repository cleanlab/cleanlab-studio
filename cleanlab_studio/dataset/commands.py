import click
from cleanlab_studio.dataset.schema import schema
from cleanlab_studio.dataset.upload import upload


@click.group(help="upload datasets, generate & validate dataset schema")
def dataset() -> None:
    pass


dataset.add_command(schema)
dataset.add_command(upload)
