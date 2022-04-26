import click
from cleanlab_cli.dataset.commands import dataset
from cleanlab_cli.login.login import login
from cleanlab_cli.settings import CleanlabSettings


@click.group()
def cli():
    CleanlabSettings.init_cleanlab_settings()
    pass


cli.add_command(login)
cli.add_command(dataset)
