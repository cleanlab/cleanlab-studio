import os

import pytest

from cleanlab_studio import Studio
from cleanlab_studio.studio.trustworthy_language_model import TLM
from cleanlab_studio.internal.tlm.concurrency import TlmRateHandler


@pytest.fixture
def tlm() -> TLM:
    """Creates a TLM with default settings."""
    try:
        # uses environment API key
        return Studio(None).TLM()
    except Exception as e:
        environment = os.environ.get("CLEANLAB_API_BASE_URL")
        pytest.skip(
            f"Failed to create TLM: {e}. Check your API key and environment: ({environment})."
        )


@pytest.fixture
def tlm_rate_handler() -> TlmRateHandler:
    """Creates a TlmRateHandler with default settings."""
    return TlmRateHandler()
