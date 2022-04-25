import json
import os
from config import PACKAGE_VERSION

VERSION_ = "version"
API_KEY_ = "api_key"


def create_settings_dict(api_key):
    return {API_KEY_: api_key, VERSION_: PACKAGE_VERSION}


def get_cleanlab_dir():
    XDG_CONFIG_HOME = os.environ.get("XDG_CONFIG_HOME", None)
    CONFIG_HOME = XDG_CONFIG_HOME if XDG_CONFIG_HOME else os.environ.get("HOME")
    cleanlab_dir = os.path.expanduser(os.path.join(CONFIG_HOME, ".cleanlab"))
    return cleanlab_dir


def get_settings_filepath():
    return os.path.join(get_cleanlab_dir(), "settings.json")


def get_cleanlab_settings():
    settings_filepath = get_settings_filepath()
    if not os.path.exists(settings_filepath):
        raise FileNotFoundError

    with open(settings_filepath, "r") as f:
        settings = json.load(f)

    if settings[VERSION_] != PACKAGE_VERSION:
        # lsv = latest_semantic_version, ssv = settings_semantic_version
        lsv = [int(x) for x in PACKAGE_VERSION.split(".")]
        ssv = [int(x) for x in settings[VERSION_].split(".")]
        if lsv[0] != ssv[0]:
            raise ValueError(
                "Major semantic version in settings does not match latest package version."
                " Settings file must be migrated or re-generated."
            )
        elif lsv[1] != ssv[1]:
            raise ValueError(
                "Minor semantic version in settings does not match latest package version. Settings"
                " file must be re-generated."
            )
        else:
            pass

    return settings


def save_cleanlab_settings(filepath, settings):
    with open(filepath, "w") as f:
        json.dump(settings, f)


def load_api_key():
    settings = get_cleanlab_settings()
    print(f"load_api_key got settings {settings}")
    return settings.get(API_KEY_)
