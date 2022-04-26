import click
from cleanlab_cli.api_service import validate_api_key
from cleanlab_cli.click_helpers import abort
from cleanlab_cli.settings import CleanlabSettings


class AuthConfig:
    def __init__(self):
        self.api_key = None

    def get_api_key(self):
        if self.api_key is None:
            try:
                api_key = CleanlabSettings.load().api_key
                if api_key is None or not validate_api_key(api_key):
                    raise ValueError("Invalid API key.")
                self.api_key = api_key
            except (FileNotFoundError, KeyError, ValueError):
                abort("No valid API key found. Run 'cleanlab login' before running this command.")
        return self.api_key


auth_config = click.make_pass_decorator(AuthConfig, ensure=True)
