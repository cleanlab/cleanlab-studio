import json
import os
from typing import Optional, Dict, Any
import semver

from cleanlab_studio.version import SETTINGS_VERSION, MIN_SETTINGS_VERSION, MAX_SETTINGS_VERSION
from cleanlab_studio.cli.types import CleanlabSettingsDict


class CleanlabSettings:
    def __init__(self, version: Optional[str], api_key: Optional[str]):
        self.version = version
        self.api_key = api_key

    @staticmethod
    def init_base() -> "CleanlabSettings":
        return CleanlabSettings(version=SETTINGS_VERSION, api_key=None)

    @staticmethod
    def from_dict(d: CleanlabSettingsDict) -> "CleanlabSettings":
        return CleanlabSettings(
            version=d.get("version", None),
            api_key=d.get("api_key", None),
        )

    def to_dict(self) -> CleanlabSettingsDict:
        return dict(version=self.version, api_key=self.api_key)

    @staticmethod
    def get_cleanlab_dir() -> str:
        XDG_CONFIG_HOME = os.environ.get("XDG_CONFIG_HOME")
        CONFIG_HOME = XDG_CONFIG_HOME if XDG_CONFIG_HOME else "~/.config"
        cleanlab_dir = os.path.expanduser(os.path.join(CONFIG_HOME, "cleanlab"))
        return cleanlab_dir

    @staticmethod
    def get_settings_filepath() -> str:
        return os.path.join(CleanlabSettings.get_cleanlab_dir(), "settings.json")

    @staticmethod
    def init_cleanlab_dir() -> None:
        """
        Initializes a cleanlab config directory if one does not currently exist. No-op if one already exists.
        :return:
        """
        cleanlab_dir = CleanlabSettings.get_cleanlab_dir()
        os.makedirs(cleanlab_dir, exist_ok=True)

    @staticmethod
    def init_cleanlab_settings() -> None:
        """
        Initializes and saves a base Cleanlab settings file. No-op if one already exists.
        :return:
        """
        CleanlabSettings.init_cleanlab_dir()
        settings_filepath = CleanlabSettings.get_settings_filepath()
        if not os.path.exists(settings_filepath):
            base = CleanlabSettings.init_base()
            base.save()

    @staticmethod
    def load() -> "CleanlabSettings":
        filepath = CleanlabSettings.get_settings_filepath()
        with open(filepath, "r") as f:
            settings_dict = json.load(f)
        settings = CleanlabSettings.from_dict(settings_dict)
        settings.validate_version()
        return settings

    def update_version(self) -> None:
        self.version = SETTINGS_VERSION
        self.save()

    def validate_version(self) -> None:
        if semver.compare(MIN_SETTINGS_VERSION, self.version) == 1:
            # TODO add proper settings migrations
            raise ValueError("Settings file must be migrated or re-generated.")
        elif semver.compare(MAX_SETTINGS_VERSION, self.version) == -1:
            raise ValueError(
                "CLI is not up to date with your settings version. Run 'pip install --upgrade cleanlab-studio'."
            )

    def save(self) -> None:
        with open(CleanlabSettings.get_settings_filepath(), "w") as f:
            json.dump(self.to_dict(), f)


def get_cleanlab_settings() -> "CleanlabSettings":
    settings_filepath = CleanlabSettings.get_settings_filepath()
    if not os.path.exists(settings_filepath):
        CleanlabSettings.init_cleanlab_settings()

    return CleanlabSettings.load()
