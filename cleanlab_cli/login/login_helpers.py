import json
import os
from config import PACKAGE_VERSION


def create_settings_dict(api_key):
    return {"api_key": api_key, "version": PACKAGE_VERSION}


def get_cleanlab_settings():
    cleanlab_dir = os.path.expanduser("~/.cleanlab/")
    settings_filepath = os.path.join(cleanlab_dir, "settings.json")
    if not os.path.exists(settings_filepath):
        raise FileNotFoundError

    with open(settings_filepath, "r") as f:
        settings = json.load(f)

    if settings[PACKAGE_VERSION] != PACKAGE_VERSION:
        # TODO
        pass

    return settings


def load_api_key():
    settings = get_cleanlab_settings()
    return settings.get("api_key")
