import click
from cleanlab_studio.cli.api_service import validate_api_key
from cleanlab_studio.cli.click_helpers import abort
from cleanlab_studio.cli.settings import CleanlabSettings
from typing import Optional


class AuthConfig:
    def __init__(self) -> None:
        self.api_key: Optional[str] = None

    def get_api_key(self) -> str:
        if self.api_key is None:
            try:
                api_key = CleanlabSettings.load().api_key
                if api_key is None or not validate_api_key(api_key):
                    raise ValueError("Invalid API key.")
                else:
                    self.api_key = api_key
                    return api_key
            except (FileNotFoundError, KeyError, ValueError):
                abort("No valid API key found. Run 'cleanlab login' before running this command.")
        assert self.api_key is not None
        return self.api_key


auth_config = click.make_pass_decorator(AuthConfig, ensure=True)
