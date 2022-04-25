import json
import os
from config import PACKAGE_VERSION


def create_settings_dict(api_key):
    return {"api_key": api_key, "version": PACKAGE_VERSION}


def get_cleanlab_settings():
    XDG_CONFIG_HOME = os.environ.get("XDG_CONFIG_HOME", None)
    CONFIG_HOME = XDG_CONFIG_HOME if XDG_CONFIG_HOME else os.environ.get("HOME")
    cleanlab_dir = os.path.expanduser(os.path.join(CONFIG_HOME, ".cleanlab"))
    settings_filepath = os.path.join(cleanlab_dir, "settings.json")
    if not os.path.exists(settings_filepath):
        raise FileNotFoundError

    with open(settings_filepath, "r") as f:
        settings = json.load(f)

    if settings[PACKAGE_VERSION] != PACKAGE_VERSION:
        # lsv = latest_semantic_version, ssv = settings_semantic_version
        lsv = [int(x) for x in PACKAGE_VERSION.split(".")]
        ssv = [int(x) for x in settings[PACKAGE_VERSION].split(".")]
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


def load_api_key():
    settings = get_cleanlab_settings()
    return settings.get("api_key")
