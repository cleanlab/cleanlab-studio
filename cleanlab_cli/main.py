import click
from cleanlab_cli.dataset.commands import dataset
from cleanlab_cli.login.login import login


@click.group()
def cli():
    pass


cli.add_command(login)
cli.add_command(dataset)
