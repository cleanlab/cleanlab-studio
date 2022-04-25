import os
import json
from cleanlab_cli.api_service import validate_api_key
from cleanlab_cli.click_helpers import *
from cleanlab_cli.login import login_helpers as helpers
from cleanlab_cli.login.login_helpers import API_KEY_, VERSION_


@click.command(help="authentication for Cleanlab Studio")
@click.option(
    "--key",
    "--k",
    type=str,
    prompt=True,
    help="API key for CLI uploads. You can get this from https://app.cleanlab.ai/upload.",
)
def login(key):
    cleanlab_dir = helpers.get_cleanlab_dir()
    if not os.path.exists(cleanlab_dir):
        os.mkdir(cleanlab_dir)

    # validate API key
    valid_key = validate_api_key(key)
    if not valid_key:
        abort("API key is invalid. Check https://app.cleanlab.ai/upload for your current API key.")

    # save API key
    settings = dict()
    settings_filepath = helpers.get_settings_filepath()

    if os.path.exists(settings_filepath):
        try:
            with open(settings_filepath, "r") as f:
                settings = json.load(f)
                settings[API_KEY_] = key
                # TODO should anything be done if version does not match?
        except json.decoder.JSONDecodeError:
            error("CLI settings are corrupted and could not be read.")
            overwrite = click.confirm(
                "Would you like to create a new settings file with the provided API key?",
                default=None,
            )
            if overwrite:
                settings = helpers.create_settings_dict(key)
            else:
                abort(
                    "Settings file is corrupted, unable to login. "
                    "Either re-run this command and allow a new settings file to be created, "
                    f"or manually fix the settings file at {settings_filepath}"
                )
    else:
        settings = helpers.create_settings_dict(key)

    helpers.save_cleanlab_settings(settings_filepath, settings)
    success(f"API key is valid. API key stored in {cleanlab_dir}")
