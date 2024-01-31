import click
import os
import subprocess

from cleanlab_studio.internal.api import api
from cleanlab_studio.cli.click_helpers import abort
from cleanlab_studio.cli.dataset.commands import dataset
from cleanlab_studio.cli.cleanset.commands import cleanset
from cleanlab_studio.cli.login.login import login
from cleanlab_studio.internal.settings import CleanlabSettings
from cleanlab_studio.version import __version__
from cleanlab_studio.internal.util import telemetry


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    if ctx.invoked_subcommand == "version":
        return  # avoid RTT / dependence on API to get client version
    CleanlabSettings.init_cleanlab_settings()
    valid_version = api.is_valid_client_version()
    if not valid_version:
        abort(
            "CLI is out of date and must be updated. Run 'pip install --upgrade cleanlab-studio'."
        )
    pass


@click.command(name="version", help="get version of cleanlab-studio")
@telemetry(load_api_key=True)
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
