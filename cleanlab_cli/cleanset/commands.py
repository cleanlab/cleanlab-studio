import click
from cleanlab_cli.cleanset.download import download


@click.group(help="download Cleanlab columns from cleansets")
def cleanset():
    pass


cleanset.add_command(download)
