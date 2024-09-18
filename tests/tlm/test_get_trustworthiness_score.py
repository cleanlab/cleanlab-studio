import asyncio
from typing import Any

import pytest

from cleanlab_studio.studio.trustworthy_language_model import TLM
from tests.tlm.conftest import make_text_unique
from tests.tlm.constants import (
    TEST_PROMPT,
    TEST_PROMPT_BATCH,
    TEST_RESPONSE,
    TEST_RESPONSE_BATCH,
)

test_prompt = make_text_unique(TEST_PROMPT)
test_prompt_batch = [make_text_unique(prompt) for prompt in TEST_PROMPT_BATCH]


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


def test_single_get_trustworthiness_score(tlm: TLM) -> None:
    """Tests running a single get_trustworthiness_score in the TLM.

    Expected:
    - TLM should return a single response
    - Response should be non-None
    - No exceptions are raised
    """
    # act -- run a single get_trustworthiness_score
    response = tlm.get_trustworthiness_score(test_prompt, TEST_RESPONSE)

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
    response = tlm.get_trustworthiness_score(test_prompt_batch, TEST_RESPONSE_BATCH)

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
            test_prompt_batch,
            TEST_RESPONSE_BATCH,
        )


def test_batch_try_get_trustworthiness_score(tlm: TLM) -> None:
    """Tests running a batch try get_trustworthiness_score in the TLM.

    Expected:
    - TLM should return a list of responses
    - Responses can be None or of type TLMResponse
    - No exceptions are raised
    """
    # act -- run a batch get_trustworthiness_score
    response = tlm.try_get_trustworthiness_score(
        test_prompt_batch,
        TEST_RESPONSE_BATCH,
    )

    # assert
    # - response is not None
    # - a list of responses of type TLMResponse or None is returned
    # - no exceptions are raised (implicit)
    assert response is not None
    assert isinstance(response, list)
    assert all(r is None or is_trustworthiness_score(r) for r in response)


def test_batch_try_get_trustworthiness_score_force_timeouts(tlm: TLM) -> None:
    """Tests running a batch try get_trustworthiness_score in the TLM, forcing timeouts.

    Sets timeout to 0.0001 seconds, which should force a timeout for all get_trustworthiness_scores.
    This should result in None responses for all get_trustworthiness_scores.

    Expected:
    - TLM should return a list of responses
    - Responses can be None or of type TLMResponse
    - No exceptions are raised
    """
    # arrange -- override timeout
    tlm._timeout = 0.0001

    # act -- run a batch get_trustworthiness_score
    response = tlm.try_get_trustworthiness_score(
        test_prompt_batch,
        TEST_RESPONSE_BATCH,
    )

    # assert
    # - response is not None
    # - all responses timed out and are None
    # - no exceptions are raised (implicit)
    assert response is not None
    assert isinstance(response, list)
    assert all(r is None for r in response)
