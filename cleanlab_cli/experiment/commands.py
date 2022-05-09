import click
from cleanlab_cli.experiment.download import download


@click.group(help="download labels from experiments")
def experiment():
    pass


experiment.add_command(download)
