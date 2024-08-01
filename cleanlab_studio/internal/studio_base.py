from typing import Optional

from cleanlab_studio.errors import MissingAPIKeyError, VersionError
from cleanlab_studio.internal.api import api
from cleanlab_studio.internal.settings import CleanlabSettings


class StudioBase:
    _api_key: str

    def __init__(self, api_key: Optional[str]):
        """
        Creates a Cleanlab Studio client.

        Args:
            api_key: You can find your API key on your [account page](https://app.cleanlab.ai/account) in Cleanlab Studio. Instead of specifying the API key here, you can also log in with `cleanlab login` on the command-line.

        """
        if not api.is_valid_client_version():
            raise VersionError(
                "CLI is out of date and must be updated. Run 'pip install --upgrade cleanlab-studio'."
            )
        if api_key is None:
            try:
                api_key = CleanlabSettings.load().api_key
                if api_key is None:
                    raise ValueError
            except (FileNotFoundError, KeyError, ValueError):
                raise MissingAPIKeyError(
                    "No API key found; either specify API key or log in with 'cleanlab login' first"
                )
        if not api.validate_api_key(api_key):
            raise ValueError(
                f"Invalid API key, please check if it is properly specified: {api_key}"
            )

        self._api_key = api_key
