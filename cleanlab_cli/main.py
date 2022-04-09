import click
from cleanlab_cli.dataset.commands import dataset
from cleanlab_cli.auth.auth import auth, auth_config


@click.group()
@auth_config
def cli(config):
    pass


cli.add_command(auth)
cli.add_command(dataset)
