import click
import os
import subprocess

from cleanlab_studio import api_service
from cleanlab_studio.click_helpers import abort
from cleanlab_studio.dataset.commands import dataset
from cleanlab_studio.cleanset.commands import cleanset
from cleanlab_studio.login.login import login
from cleanlab_studio.settings import CleanlabSettings
from cleanlab_studio.version import __version__


@click.group()
def cli() -> None:
    CleanlabSettings.init_cleanlab_settings()
    valid_version = api_service.check_client_version()
    if not valid_version:
        abort(
            "CLI is out of date and must be updated. Run 'pip install --upgrade cleanlab-studio'."
        )
    pass


@click.command(name="version", help="get version of cleanlab-studio")
def version() -> None:
    # detect if user has installed the package from source in editable mode
    try:
        dir = os.path.dirname(__file__)
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=dir, stderr=subprocess.DEVNULL
        )[:10].decode("utf8")
        dirty = subprocess.check_output(["git", "diff", "--stat"], stderr=subprocess.DEVNULL) != b""
        print(f"cleanlab-studio {__version__} (git sha1 {sha}{'-dirty' if dirty else ''})")
    except:
        print(f"cleanlab-studio {__version__}")


cli.add_command(login)
cli.add_command(dataset)
cli.add_command(cleanset)
cli.add_command(version)
