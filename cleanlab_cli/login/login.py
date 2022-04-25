import os
import json
from config import PACKAGE_VERSION
from cleanlab_cli.api_service import validate_api_key
from cleanlab_cli.click_helpers import *


class AuthConfig:
    def __init__(self):
        self.api_key = None

    def get_api_key(self):
        if self.api_key is None:
            try:
                api_key = load_api_key()
                if api_key is None or not validate_api_key(api_key):
                    raise ValueError("Invalid API key.")
                self.api_key = api_key
            except (FileNotFoundError, KeyError, ValueError):
                abort("No valid API key found. Run 'cleanlab login' before running this command.")
        return self.api_key


auth_config = click.make_pass_decorator(AuthConfig, ensure=True)


def create_settings_dict(api_key):
    return {"api_key": api_key, "version": PACKAGE_VERSION}


def get_cleanlab_settings():
    cleanlab_dir = os.path.expanduser("~/.cleanlab/")
    settings_filepath = os.path.join(cleanlab_dir, "settings.json")
    if not os.path.exists(settings_filepath):
        raise FileNotFoundError

    with open(settings_filepath, "r") as f:
        settings = json.load(f)
    return settings


def load_api_key():
    settings = get_cleanlab_settings()
    return settings.get("api_key")


@click.command(help="authentication for Cleanlab Studio")
@click.option(
    "--key",
    "--k",
    type=str,
    prompt=True,
    help="API key for CLI uploads. You can get this from https://app.cleanlab.ai/upload.",
)
def login(key):
    cleanlab_dir = os.path.expanduser("~/.cleanlab/")
    if not os.path.exists(cleanlab_dir):
        os.mkdir(cleanlab_dir)

    # validate API key
    valid_key = validate_api_key(key)
    if not valid_key:
        abort("API key is invalid. Check https://app.cleanlab.ai/upload for your current API key.")

    # save API key
    settings = dict()
    settings_filepath = os.path.join(cleanlab_dir, "settings.json")
    if os.path.exists(settings_filepath):
        try:
            with open(settings_filepath, "r") as f:
                settings = json.load(f)
                settings["api_key"] = key
        except json.decoder.JSONDecodeError:
            error("CLI settings are corrupted and could not be read.")
            overwrite = click.confirm(
                "Would you like to create a new settings file with the provided API key?",
                default=None,
            )
            if overwrite:
                settings = create_settings_dict(key)
            else:
                abort(
                    "Settings file is corrupted, unable to login. "
                    "Either re-run this command and allow a new settings file to be created, "
                    f"or manually fix the settings file at {settings_filepath}"
                )
    else:
        settings = create_settings_dict(key)

    with open(settings_filepath, "w") as f:
        json.dump(settings, f)

    success(f"API key is valid. API key stored in {cleanlab_dir}")
