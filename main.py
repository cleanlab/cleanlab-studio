import click
from dataset.dataset import dataset
from auth.auth import login, auth_config


@click.group()
@auth_config
def cli(config):
    pass


cli.add_command(login)
cli.add_command(dataset)
