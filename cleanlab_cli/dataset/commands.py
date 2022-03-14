import click
from cleanlab_cli.auth.auth import auth_config
from cleanlab_cli.dataset.schema import schema
from cleanlab_cli.dataset.upload import upload


@click.group()
@auth_config
def dataset(config):
    pass


dataset.add_command(schema)
dataset.add_command(upload)
