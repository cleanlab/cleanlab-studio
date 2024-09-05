import os
from typing import Any, Dict, Optional

import numpy as np
import pytest

from cleanlab_studio import Studio
from cleanlab_studio.internal.constants import (
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
    Save randomly created options dictionary for each tlm object as well."""

    tlm_dict = {}
    for quality_preset in _VALID_TLM_QUALITY_PRESETS:
        tlm_dict[quality_preset] = {}
        for model in _VALID_TLM_MODELS + [None]:
            options = _get_options_dictionary(model)
            tlm_dict[quality_preset][model]["tlm"] = studio.TLM(
                quality_preset=quality_preset, options=options
            )
            tlm_dict[quality_preset][model]["options"] = options
    return tlm_dict


@pytest.fixture
def tlm_rate_handler() -> TlmRateHandler:
    """Creates a TlmRateHandler with default settings."""
    return TlmRateHandler()


def _get_options_dictionary(model: Optional[str]) -> dict:
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
        options["max_tokens"] = np.random.randint(64, 512)
    if add_use_self_reflection:
        options["use_self_reflection"] = np.random.choice([True, False])
    if add_num_candidate_responses:
        options["num_candidate_responses"] = np.random.randint(1, 5)
    if add_num_consistency_samples:
        options["num_consistency_samples"] = np.random.randint(0, 10)

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
