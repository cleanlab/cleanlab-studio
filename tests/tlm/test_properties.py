import asyncio
from typing import Any, Dict, List, Union

import pytest

from cleanlab_studio.internal.constants import _VALID_TLM_QUALITY_PRESETS
from cleanlab_studio.studio.trustworthy_language_model import TLM
from tests.tlm.conftest import make_text_unique
from tests.tlm.constants import (
    MODELS_WITH_NO_PERPLEXITY_SCORE,
    TEST_PROMPT,
    TEST_PROMPT_BATCH,
    TEST_RESPONSE,
    TEST_RESPONSE_BATCH,
    VALID_TLM_MODELS,
)
from tests.tlm.test_get_trustworthiness_score import (
    is_trustworthiness_score_json_format,
)
from tests.tlm.test_prompt import is_tlm_response

test_prompt_single = make_text_unique(TEST_PROMPT)
test_prompt_batch = [make_text_unique(prompt) for prompt in TEST_PROMPT_BATCH]


def _test_log(response: Dict[str, Any], options: Dict[str, Any]) -> None:
    """Tests the log dictionary in the response based on the options dictionary."""
    if "log" in options.keys():
        print("Testing log:", options["log"], end="")
        if "log" in response:
            print(" response log:", response["log"], end="")
        else:
            print("... FAILED. NO LOG IN RESPONSE.")
        assert "log" in response
        assert isinstance(response["log"], dict)
        if "perplexity" in options["log"]:
            assert (
                isinstance(response["log"]["perplexity"], float)
                or response["log"]["perplexity"] is None
            )
        if "explanation" in options["log"]:
            assert isinstance(response["log"]["explanation"], str)
        print("... PASSED.")


def _test_log_batch(responses: Dict[str, Any], options: Dict[str, Any]) -> None:
    """Tests the log dictionary in the batch response based on the options dictionary."""
    for response in responses:
        if response is not None:
            _test_log(response, options)


def _is_valid_prompt_response(
    response,
    options,
    allow_none_response=False,
    allow_null_trustworthiness_score=False,
) -> bool:
    """Returns true if prompt response is valid based on properties for prompt() functionality."""
    _test_log(response, options)
    if {"use_self_reflection", "num_consistency_samples"}.issubset(options) and (
        options["num_consistency_samples"] == 0 and not options["use_self_reflection"]
    ):
        print("Options dictinary called with strange parameters. Allowing none in response.")
        return is_tlm_response(
            response,
            allow_none_response=allow_none_response,
            allow_null_trustworthiness_score=True,
        )
    else:
        return is_tlm_response(
            response,
            allow_none_response=allow_none_response,
            allow_null_trustworthiness_score=allow_null_trustworthiness_score,
        )


def _is_valid_get_trustworthiness_score_response(
    response,
    options,
    allow_null_trustworthiness_score=False,
) -> bool:
    """Returns true if trustworthiness score is valid based on properties for get_trustworthiness_score() functionality."""
    assert isinstance(response, dict)

    if (
        ({"quality_preset", "use_self_reflection"}.issubset(options))
        and not options["use_self_reflection"]
        and options["quality_preset"] == "base"
    ):
        print("Options dictinary called with strange parameters. Allowing none in response.")
        return is_trustworthiness_score_json_format(response, allow_null_trustworthiness_score=True)
    elif (
        ({"num_consistency_samples", "use_self_reflection"}.issubset(options))
        and not options["use_self_reflection"]
        and options["num_consistency_samples"] == 0
    ):
        print("Options dictinary called with strange parameters. Allowing none in response.")
        return is_trustworthiness_score_json_format(response, allow_null_trustworthiness_score=True)
    else:
        return is_trustworthiness_score_json_format(
            response,
            allow_null_trustworthiness_score=allow_null_trustworthiness_score,
        )


def _test_prompt_response(
    response,
    options,
    allow_none_response=False,
    allow_null_trustworthiness_score=False,
) -> None:
    """Property tests the responses of a prompt based on the options dictionary and returned responses."""
    assert _is_valid_prompt_response(
        response=response,
        options=options,
        allow_none_response=allow_none_response,
        allow_null_trustworthiness_score=allow_null_trustworthiness_score,
    )


def _test_batch_prompt_response(
    responses,
    options,
    allow_none_response=False,
    allow_null_trustworthiness_score=False,
) -> None:
    """Property tests the responses of a batch prompt based on the options dictionary and returned responses."""
    assert responses is not None
    assert isinstance(responses, list)
    _test_log_batch(responses, options)

    checked_responses = [
        _is_valid_prompt_response(
            response,
            options,
            allow_none_response=allow_none_response,
            allow_null_trustworthiness_score=allow_null_trustworthiness_score,
        )
        for response in responses
    ]
    print("Checked respones:", checked_responses)
    assert all(checked_responses)


def _test_get_trustworthiness_score_response(
    response,
    options,
    allow_null_trustworthiness_score=False,
) -> None:
    """Property tests the responses of a get_trustworthiness_score based on the options dictionary and returned responses."""
    assert _is_valid_get_trustworthiness_score_response(
        response=response,
        options=options,
        allow_null_trustworthiness_score=allow_null_trustworthiness_score,
    )


def _test_batch_get_trustworthiness_score_response(
    responses, options, allow_null_trustworthiness_score=False
) -> None:
    """Property tests the responses of a batch get_trustworthiness_score based on the options dictionary and returned responses."""
    assert responses is not None
    assert isinstance(responses, list)
    _test_log_batch(responses, options)

    checked_responses = [
        _is_valid_get_trustworthiness_score_response(
            response,
            options,
            allow_null_trustworthiness_score=allow_null_trustworthiness_score,
        )
        for response in responses
    ]
    print("Checked respones:", checked_responses)
    assert all(checked_responses)


@pytest.mark.asyncio(scope="function")
async def _run_prompt_async(tlm: TLM, prompt: Union[List[str], str]) -> Any:
    """Runs tlm.prompt() asynchronously."""
    return await tlm.prompt_async(prompt)


@pytest.mark.asyncio(scope="function")
async def _run_get_trustworthiness_score_async(
    tlm: TLM, prompt: Union[List[str], str], response: Union[List[str], str]
) -> Any:
    """Runs tlm.get_trustworthiness_score asynchronously."""
    return await tlm.get_trustworthiness_score_async(prompt, response)


@pytest.mark.parametrize("model", VALID_TLM_MODELS)
@pytest.mark.parametrize("quality_preset", _VALID_TLM_QUALITY_PRESETS)
def test_prompt(tlm_dict: Dict[str, Any], model: str, quality_preset: str) -> None:
    """Tests running a prompt in the TLM for all quality_presets, model types and single/batch prompt."""
    print("Testing with prompt:", test_prompt_single)
    print("Testing with batch prompt:", test_prompt_batch)
    # get TLMs and options dictionary based on parameters
    tlm = tlm_dict[quality_preset][model]["tlm"]
    tlm_no_options = tlm_dict[quality_preset][model]["tlm_no_options"]
    options = tlm_dict[quality_preset][model]["options"]
    allow_null_trustworthiness_score = (
        quality_preset == "base" and model in MODELS_WITH_NO_PERPLEXITY_SCORE
    )
    print("TLM with no options called on single query run.")
    print("TLM Options for run:", options)

    # test prompt with single prompt
    response = tlm_no_options.prompt(test_prompt_single)
    print("TLM Single Response:", response)
    _test_prompt_response(
        response,
        {},
        allow_null_trustworthiness_score=allow_null_trustworthiness_score,
    )

    # test prompt with batch prompt
    responses = tlm.prompt(test_prompt_batch)
    print("TLM Batch Responses:", responses)
    _test_batch_prompt_response(
        responses,
        options,
        allow_null_trustworthiness_score=allow_null_trustworthiness_score,
    )


@pytest.mark.parametrize("model", VALID_TLM_MODELS)
@pytest.mark.parametrize("quality_preset", _VALID_TLM_QUALITY_PRESETS)
def test_prompt_async(tlm_dict: Dict[str, Any], model: str, quality_preset: str) -> None:
    """Tests running a prompt_async in the TLM for all quality_presets, model types and single/batch prompt."""
    print("Testing with prompt:", test_prompt_single)
    print("Testing with batch prompt:", test_prompt_batch)
    # get TLMs and options dictionary based on parameters
    tlm = tlm_dict[quality_preset][model]["tlm"]
    tlm_no_options = tlm_dict[quality_preset][model]["tlm_no_options"]
    options = tlm_dict[quality_preset][model]["options"]
    allow_null_trustworthiness_score = (
        quality_preset == "base" and model in MODELS_WITH_NO_PERPLEXITY_SCORE
    )
    print("TLM with no options called on single query run.")
    print("TLM Options for run:", options)

    # test prompt with single prompt
    response = asyncio.run(_run_prompt_async(tlm_no_options, test_prompt_single))
    print("TLM Single Response:", response)
    _test_prompt_response(
        response, {}, allow_null_trustworthiness_score=allow_null_trustworthiness_score
    )

    # test prompt with batch prompt
    responses = asyncio.run(_run_prompt_async(tlm, test_prompt_batch))
    print("TLM Batch Responses:", responses)
    _test_batch_prompt_response(
        responses,
        options,
        allow_null_trustworthiness_score=allow_null_trustworthiness_score,
    )


@pytest.mark.parametrize("model", VALID_TLM_MODELS)
@pytest.mark.parametrize("quality_preset", _VALID_TLM_QUALITY_PRESETS)
def test_try_prompt(tlm_dict: Dict[str, Any], model: str, quality_preset: str) -> None:
    """Tests running try_prompt in the TLM for all quality_presets, model types batch prompt."""
    print("Testing with batch prompt:", test_prompt_batch)
    # get TLM and options dictionary based on parameters
    tlm_no_options = tlm_dict[quality_preset][model]["tlm_no_options"]
    options = tlm_dict[quality_preset][model]["options"]
    allow_null_trustworthiness_score = (
        quality_preset == "base" and model in MODELS_WITH_NO_PERPLEXITY_SCORE
    )
    print("TLM with no options called on batch query run.")
    print("TLM Options for run: None.")

    # test prompt with batch prompt
    responses = tlm_no_options.try_prompt(test_prompt_batch)
    print("TLM Batch Responses:", responses)
    _test_batch_prompt_response(
        responses,
        {},
        allow_none_response=True,
        allow_null_trustworthiness_score=allow_null_trustworthiness_score,
    )


@pytest.mark.parametrize("model", VALID_TLM_MODELS)
@pytest.mark.parametrize("quality_preset", _VALID_TLM_QUALITY_PRESETS)
def test_get_trustworthiness_score(
    tlm_dict: Dict[str, Any], model: str, quality_preset: str
) -> None:
    """Tests running get_trustworthiness_score in the TLM for all quality_presets, model types and single/batch prompt."""
    print("Testing with prompt/response:", test_prompt_single, TEST_RESPONSE)
    print("Testing with batch prompt/response:", test_prompt_batch, TEST_RESPONSE_BATCH)
    # get TLMs and options dictionary based on parameters
    tlm = tlm_dict[quality_preset][model]["tlm"]
    tlm_no_options = tlm_dict[quality_preset][model]["tlm_no_options"]
    options = tlm_dict[quality_preset][model]["options"]
    print("TLM with no options called on batch query run.")
    print("TLM Options for run:", options)

    # test prompt with single prompt
    response = tlm.get_trustworthiness_score(test_prompt_single, TEST_RESPONSE)
    print("TLM Single Response:", response)
    _test_get_trustworthiness_score_response(response, options)

    # test prompt with batch prompt
    responses = tlm_no_options.get_trustworthiness_score(test_prompt_batch, TEST_RESPONSE_BATCH)
    print("TLM Batch Responses:", responses)
    _test_batch_get_trustworthiness_score_response(responses, {})


@pytest.mark.parametrize("model", VALID_TLM_MODELS)
@pytest.mark.parametrize("quality_preset", _VALID_TLM_QUALITY_PRESETS)
def test_get_trustworthiness_score_async(
    tlm_dict: Dict[str, Any], model: str, quality_preset: str
) -> None:
    """Tests running get_trustworthiness_score_async in the TLM for all quality_presets, model types and single/batch prompt."""
    print("Testing with prompt/response:", test_prompt_single, TEST_RESPONSE)
    print("Testing with batch prompt/response:", test_prompt_batch, TEST_RESPONSE_BATCH)
    # get TLMs and options dictionary based on parameters
    tlm = tlm_dict[quality_preset][model]["tlm"]
    tlm_no_options = tlm_dict[quality_preset][model]["tlm_no_options"]
    options = tlm_dict[quality_preset][model]["options"]
    print("TLM with no options called on single query run.")
    print("TLM Options for run:", options)

    # test prompt with single prompt
    response = asyncio.run(
        _run_get_trustworthiness_score_async(tlm_no_options, test_prompt_single, TEST_RESPONSE)
    )
    print("TLM Single Response:", response)
    _test_get_trustworthiness_score_response(response, {})

    # test prompt with batch prompt
    responses = asyncio.run(
        _run_get_trustworthiness_score_async(
            tlm,
            test_prompt_batch,
            TEST_RESPONSE_BATCH,
        )
    )
    print("TLM Batch Responses:", responses)
    _test_batch_get_trustworthiness_score_response(responses, options)


@pytest.mark.parametrize("model", VALID_TLM_MODELS)
@pytest.mark.parametrize("quality_preset", _VALID_TLM_QUALITY_PRESETS)
def test_try_get_trustworithness_score(
    tlm_dict: Dict[str, Any], model: str, quality_preset: str
) -> None:
    """Tests running try_get_trustworthiness_score in the TLM for all quality_presets, model types and batch prompt."""
    print("Testing with batch prompt/response:", test_prompt_batch, TEST_RESPONSE_BATCH)
    # get TLMs and options dictionary based on parameters
    tlm = tlm_dict[quality_preset][model]["tlm"]
    options = tlm_dict[quality_preset][model]["options"]
    print("TLM without options is not called.")
    print("TLM Options for run:", options)

    # test prompt with batch prompt
    responses = tlm.try_get_trustworthiness_score(test_prompt_batch, TEST_RESPONSE_BATCH)
    print("TLM Batch Responses:", responses)
    _test_batch_get_trustworthiness_score_response(responses, options)
