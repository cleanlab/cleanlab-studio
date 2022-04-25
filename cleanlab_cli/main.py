import click
from cleanlab_cli.dataset.commands import dataset
from cleanlab_cli.login.login import login, auth_config


@click.group()
@auth_config
def cli(config):
    pass


cli.add_command(login)
cli.add_command(dataset)
