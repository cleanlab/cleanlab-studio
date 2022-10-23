import click
from cleanlab_studio.cleanset.download import download


@click.group(help="download Cleanlab columns from cleansets")
def cleanset() -> None:
    pass


cleanset.add_command(download)
