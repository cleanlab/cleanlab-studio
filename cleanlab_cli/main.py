import click

from cleanlab_cli import api_service
from cleanlab_cli.click_helpers import abort
from cleanlab_cli.dataset.commands import dataset
from cleanlab_cli.cleanset.commands import cleanset
from cleanlab_cli.login.login import login
from cleanlab_cli.settings import CleanlabSettings


@click.group()
def cli():
    CleanlabSettings.init_cleanlab_settings()
    valid_version = api_service.check_client_version()
    if not valid_version:
        abort("CLI is out of date and must be updated. Run 'pip install --upgrade cleanlab-cli'.")
    pass


cli.add_command(login)
cli.add_command(dataset)
cli.add_command(cleanset)
