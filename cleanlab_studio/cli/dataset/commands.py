import click
from cleanlab_studio.cli.dataset.upload import upload


@click.group(help="upload datasets, generate & validate dataset schema")
def dataset() -> None:
    pass


dataset.add_command(upload)
