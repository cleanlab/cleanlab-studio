import click
from cleanlab_cli.cleanset.download import download


@click.group(help="download labels from experiments")
def cleanset():
    pass


cleanset.add_command(download)
