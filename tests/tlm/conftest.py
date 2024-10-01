import os
import random
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pytest

from cleanlab_studio import Studio
from cleanlab_studio.internal.constants import (
    _TLM_DEFAULT_MODEL,
    _TLM_MAX_TOKEN_RANGE,
    _VALID_TLM_MODELS,
    _VALID_TLM_QUALITY_PRESETS,
)
from cleanlab_studio.internal.tlm.concurrency import TlmRateHandler
from cleanlab_studio.studio.trustworthy_language_model import TLM


@pytest.fixture(scope="module")
def studio() -> Studio:
    """Creates a Studio with default settings."""

    try:
        # uses environment API key
        return Studio(None)
    except Exception as e:
        environment = os.environ.get("CLEANLAB_API_BASE_URL")
        pytest.skip(
            f"Failed to create Studio: {e}. Check your API key and environment: ({environment})."
        )


@pytest.fixture(scope="module")
def tlm(studio: Studio) -> TLM:
    """Creates a TLM with default settings."""
    return studio.TLM()


@pytest.fixture(scope="module")
def tlm_dict(studio: Studio) -> Dict[str, Any]:
    """Creates a dictionary of initialized tlm objects for each quality preset and model to be reused throughout the test.
    Save randomly created options dictionary for each tlm object as well.

    Initializes two TLM objects for each quality preset and model:
    - One with randomly generated options
    - One with default presets (no options)

    Each function call is tested on both of these TLM objects to ensure that the function works with options and for the default preset
    and to give signal if the function is not working for a specific set of options or overall.
    """

    tlm_dict = {}
    for quality_preset in _VALID_TLM_QUALITY_PRESETS:
        tlm_dict[quality_preset] = {}
        for model in _VALID_TLM_MODELS:
            tlm_dict[quality_preset][model] = {}
            options = _get_options_dictionary(model)
            tlm_dict[quality_preset][model]["tlm"] = studio.TLM(
                quality_preset=quality_preset, options=options
            )
            tlm_dict[quality_preset][model]["tlm_no_options"] = studio.TLM(
                quality_preset=quality_preset,
            )
            options["quality_preset"] = quality_preset
            tlm_dict[quality_preset][model]["options"] = options
    return tlm_dict


@pytest.fixture
def tlm_rate_handler() -> TlmRateHandler:
    """Creates a TlmRateHandler with default settings."""
    return TlmRateHandler()


def _get_options_dictionary(model: Optional[str]) -> dict:
    """Returns a dictionary of randomly generated options for the TLM."""

    if model is None:
        options = {}
    else:
        options = {"model": model}

    add_max_tokens = np.random.choice([True, False])
    add_num_candidate_responses = np.random.choice([True, False])
    add_num_consistency_samples = np.random.choice([True, False])
    add_use_self_reflection = np.random.choice([True, False])
    add_log_explanation = np.random.choice([True, False])
    add_log_perplexity_score = np.random.choice([True, False])

    if add_max_tokens:
        max_tokens = _TLM_MAX_TOKEN_RANGE[options.get("model", _TLM_DEFAULT_MODEL)][1]
        options["max_tokens"] = int(np.random.randint(64, max_tokens))
    if add_use_self_reflection:
        options["use_self_reflection"] = random.choice([True, False])
    if add_num_candidate_responses:
        options["num_candidate_responses"] = int(np.random.randint(1, 5))
    if add_num_consistency_samples:
        options["num_consistency_samples"] = int(np.random.randint(0, 10))

    if add_log_explanation or add_log_perplexity_score:
        options["log"] = [
            key
            for key, options_flag in {
                "explanation": add_log_explanation,
                "perplexity": add_log_perplexity_score,
            }.items()
            if options_flag
        ]
    return options


def make_text_unique(text: str) -> str:
    """Makes a text unique by prepending the curent datatime to it."""
    return str(datetime.now().strftime("%Y%m%d%H%M%S")) + " " + text
