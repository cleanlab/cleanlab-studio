import json
from cleanlab_studio.cli.api_service import validate_api_key
from cleanlab_studio.cli.click_helpers import *
from cleanlab_studio.cli.decorators.previous_state import PreviousState
from cleanlab_studio.cli.decorators import previous_state
from cleanlab_studio.internal.settings import CleanlabSettings
import click
from cleanlab_studio.internal.util import telemetry
from cleanlab_studio.errors import AuthError


@click.command(help="authentication for Cleanlab Studio")
@click.option(
    "--key",
    "--k",
    type=str,
    prompt=True,
    help="API key for CLI uploads. You can get this from https://app.cleanlab.ai/upload.",
)
@previous_state
@telemetry(load_api_key=True)
def login(prev_state: PreviousState, key: str) -> None:
    prev_state.init_state(dict(command="login", args=dict(key=key)))
    CleanlabSettings.init_cleanlab_dir()

    # validate API key
    valid_key = validate_api_key(key)
    if not valid_key:
        raise AuthError()

    # save API key
    try:
        settings = CleanlabSettings.load()
        settings.api_key = key
        settings.save()

    except json.decoder.JSONDecodeError:
        error("CLI settings are corrupted and could not be read.")
        overwrite = click.confirm(
            "Create a new settings file with the provided API key?",
            default=None,
        )
        if overwrite:
            base = CleanlabSettings.init_base()
            base.api_key = key
            base.save()
        else:
            abort(
                "Settings file is corrupted, unable to login. "
                "Either re-run this command and allow a new settings file to be created, "
                f"or manually fix the settings file at {CleanlabSettings.get_settings_filepath()}"
            )

    success(f"API key is valid. API key stored in {CleanlabSettings.get_cleanlab_dir()}")
