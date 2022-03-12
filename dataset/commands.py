import click
from auth.auth import auth_config
from dataset.schema import schema
from dataset.upload import upload


@click.group()
@auth_config
def dataset(config):
    pass


dataset.add_command(schema)
dataset.add_command(upload)
