import asyncio
from typing import Any, Dict, Optional

import pytest

from cleanlab_studio.internal.constants import (
    _VALID_TLM_MODELS,
    _VALID_TLM_QUALITY_PRESETS,
)
from tests.tlm.test_prompt import is_tlm_response


def _test_log(response: Dict[str, Any], options: Dict[str, Any]) -> None:
    """Tests the log dictionary in the response based on the options dictionary."""
    if "log" in options.keys():
        assert isinstance(response["log"], dict)
        if "perplexity" in options["log"]:
            assert isinstance(response["log"]["perplexity"], float)
        if "explanation" in options["log"]:
            assert isinstance(response["log"]["explanation"], str)


def _test_log_batch(responses: Dict[str, Any], options: Dict[str, Any]) -> None:
    """Tests the log dictionary in the batch response based on the options dictionary."""
    for response in responses:
        _test_log(response, options)


@pytest.mark.parametrize("model", _VALID_TLM_MODELS)
@pytest.mark.parametrize("quality_preset", _VALID_TLM_QUALITY_PRESETS)
def test_prompt(tlm_dict: Dict[str, Any], model: str, quality_preset: str) -> None:
    """Tests running a prompt in the TLM for all quality_presets, model types and single/batch prompt."""
    # get TLM and options dictionary based on parameters
    tlm = tlm_dict[quality_preset][model]["tlm"]
    options = tlm_dict[quality_preset][model]["options"]

    # test prompt with single prompt
    response = tlm.prompt("What is the capital of France?")
    assert response is not None
    assert is_tlm_response(response)
    _test_log(response, options)

    # test prompt with batch prompt
    responses = tlm.prompt(["What is the capital of France?", "What is the capital of Ukraine?"])
    print(responses)
    assert responses is not None
    assert isinstance(responses, list)
    assert all(is_tlm_response(r) for r in responses)
    _test_log_batch(responses, options)


# @pytest.mark.parametrize("model", _VALID_TLM_MODELS)
# @pytest.mark.parametrize("quality_preset", _VALID_TLM_QUALITY_PRESETS)
# def test_prompt_async(tlm_dict: Dict[str, Any], model: str, quality_preset: str) -> None:
#     """Tests running a prompt in the TLM for all quality_presets, model types and single/batch prompt."""
#     # get TLM and options dictionary based on parameters
#     tlm = tlm_dict[quality_preset][model]["tlm"]
#     options = tlm_dict[quality_preset][model]["options"]

#     # test prompt with single prompt
#     response = tlm.prompt_async("What is the capital of France?")
#     assert response is not None
#     assert is_tlm_response(response)
#     _test_log(response, options)

#     # test prompt with batch prompt
#     responses = tlm.prompt_async(
#         ["What is the capital of France?", "What is the capital of Ukraine?"]
#     )

#     assert responses is not None
#     assert isinstance(responses, list)
#     assert all(is_tlm_response(r) for r in response)
#     _test_log_batch(responses, options)
