import click
from cleanlab_cli.dataset.schema import schema
from cleanlab_cli.dataset.upload import upload


@click.group(help="upload datasets, generate & validate dataset schema")
def dataset():
    pass


dataset.add_command(schema)
dataset.add_command(upload)
