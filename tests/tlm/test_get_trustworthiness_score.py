import asyncio
from typing import Any

import pytest

from cleanlab_studio.studio.trustworthy_language_model import TLM


def is_trustworthiness_score(response: Any) -> bool:
    """Returns True if the response is a trustworthiness score with valid range."""
    if isinstance(response, float):
        return 0.0 <= response <= 1.0
    elif (
        isinstance(response, dict)
        and "trustworthiness_score" in response
        and isinstance(response["trustworthiness_score"], float)
    ):
        return 0.0 <= response["trustworthiness_score"] <= 1.0
    else:
        return False


def is_trustworthiness_score_json_format(response: Any) -> bool:
    """Returns True if the response is a trustworthiness score in JSON format."""
    return (
        isinstance(response, dict)
        and "trustworthiness_score" in response
        and isinstance(response["trustworthiness_score"], float)
    )


def is_valid_tlm_score_response_with_error(response: Any) -> bool:
    """Validates if the response matches the expected TLMScore with error format."""
    return (
        isinstance(response, dict)
        and "trustworthiness_score" in response
        and response["trustworthiness_score"] is None
        and (
            isinstance(response["log"], dict)
            and "error" in response["log"]
            and isinstance(response["log"]["error"], dict)
            and "message" in response["log"]["error"]
            and "retryable" in response["log"]["error"]
            and isinstance(response["log"]["error"]["retryable"], bool)
        )
    )


def test_single_get_trustworthiness_score(tlm: TLM) -> None:
    """Tests running a single get_trustworthiness_score in the TLM.

    Expected:
    - TLM should return a single response
    - Response should be non-None
    - No exceptions are raised
    """
    # act -- run a single get_trustworthiness_score
    response = tlm.get_trustworthiness_score("What is the capital of France?", "Paris")

    # assert
    # - response is not None
    # - a single response of type TLMResponse is returned
    # - no exceptions are raised (implicit)
    assert response is not None
    assert is_trustworthiness_score(response)


def test_batch_get_trustworthiness_score(tlm: TLM) -> None:
    """Tests running a batch get_trustworthiness_score in the TLM.

    Expected:
    - TLM should return a list of responses
    - Responses should be non-None
    - No exceptions are raised
    - Each response should be of type TLMResponse
    """
    # act -- run a batch get_trustworthiness_score
    response = tlm.get_trustworthiness_score(
        ["What is the capital of France?"] * 3,
        ["Paris"] * 3,
    )

    # assert
    # - response is not None
    # - a list of responses of type TLMResponse is returned
    # - no exceptions are raised (implicit)
    assert response is not None
    assert isinstance(response, list)
    assert all(is_trustworthiness_score(r) for r in response)


def test_batch_get_trustworthiness_score_force_timeouts(tlm: TLM) -> None:
    """Tests running a batch get_trustworthiness_score in the TLM, forcing timeouts.

    Sets timeout to 0.0001 seconds, which should force a timeout for all get_trustworthiness_scores.
    This should result in a timeout error being thrown

    Expected:
    - TLM should raise a timeout error
    """
    # arrange -- override timeout
    tlm._timeout = 0.0001

    # assert -- timeout is thrown
    with pytest.raises(asyncio.TimeoutError):
        # act -- run a batch get_trustworthiness_score
        tlm.get_trustworthiness_score(
            ["What is the capital of France?"] * 3,
            ["Paris"] * 3,
        )


def test_batch_try_get_trustworthiness_score(tlm: TLM) -> None:
    """Tests running a batch try get_trustworthiness_score in the TLM.

    Expected:
    - TLM should return a list of responses
    - Responses will be of type TLMResponse
    - No exceptions are raised
    """
    # act -- run a batch get_trustworthiness_score
    response = tlm.try_get_trustworthiness_score(
        ["What is the capital of France?"] * 3,
        ["Paris"] * 3,
    )

    # assert
    # - response is not None
    # - a list of responses of type TLMResponse or None is returned
    # - no exceptions are raised (implicit)
    assert response is not None
    assert isinstance(response, list)
    assert all(is_trustworthiness_score_json_format(r) for r in response)


def test_batch_try_get_trustworthiness_score_force_timeouts(tlm: TLM) -> None:
    """Tests running a batch try get_trustworthiness_score in the TLM, forcing timeouts.

    Sets timeout to 0.0001 seconds, which should force a timeout for all get_trustworthiness_scores.
    This should result in None responses for all get_trustworthiness_scores.

    Expected:
    - TLM should return a list of responses
    - Responses will be of type TLMResponse
    - No exceptions are raised
    """
    # arrange -- override timeout
    tlm._timeout = 0.0001

    # act -- run a batch get_trustworthiness_score
    response = tlm.try_get_trustworthiness_score(
        ["What is the capital of France?"] * 3,
        ["Paris"] * 3,
    )

    # assert
    # - response is not None
    # - all responses timed out and are None
    # - no exceptions are raised (implicit)
    assert response is not None
    assert isinstance(response, list)
    assert all(is_valid_tlm_score_response_with_error(r) for r in response)
