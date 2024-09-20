from typing import Any

import numpy as np
import pytest

from cleanlab_studio.errors import TlmBadRequest, ValidationError
from cleanlab_studio.studio.studio import Studio
from cleanlab_studio.studio.trustworthy_language_model import TLM
from tests.tlm.conftest import make_text_unique
from tests.tlm.constants import (
    CHARACTERS_PER_TOKEN,
    MAX_COMBINED_LENGTH_TOKENS,
    MAX_PROMPT_LENGTH_TOKENS,
    MAX_RESPONSE_LENGTH_TOKENS,
    TEST_PROMPT,
    TEST_PROMPT_BATCH,
    TEST_RESPONSE,
    TEST_RESPONSE_BATCH,
)

from .test_get_trustworthiness_score import is_tlm_score_response_with_error
from .test_prompt import is_tlm_response_with_error

np.random.seed(0)
test_prompt = make_text_unique(TEST_PROMPT)
test_prompt_batch = [make_text_unique(prompt) for prompt in TEST_PROMPT_BATCH]


def assert_prompt_too_long_error(response: Any, index: int):
    assert is_tlm_response_with_error(response)
    assert response["log"]["error"]["message"].startswith(
        f"Error executing query at index {index}:"
    )
    assert (
        "Prompt length exceeds maximum length of 70000 tokens"
        in response["log"]["error"]["message"]
    )
    assert response["log"]["error"]["retryable"] is False


def assert_prompt_too_long_error_score(response: Any, index: int):
    assert is_tlm_score_response_with_error(response)
    assert response["log"]["error"]["message"].startswith(
        f"Error executing query at index {index}:"
    )
    assert (
        "Prompt length exceeds maximum length of 70000 tokens"
        in response["log"]["error"]["message"]
    )
    assert response["log"]["error"]["retryable"] is False


def assert_response_too_long_error_score(response: Any, index: int):
    assert is_tlm_score_response_with_error(response)
    assert response["log"]["error"]["message"].startswith(
        f"Error executing query at index {index}:"
    )
    assert (
        "Response length exceeds maximum length of 15000 tokens"
        in response["log"]["error"]["message"]
    )
    assert response["log"]["error"]["retryable"] is False


def assert_prompt_and_response_combined_too_long_error_score(response: Any, index: int):
    assert is_tlm_score_response_with_error(response)
    assert response["log"]["error"]["message"].startswith(
        f"Error executing query at index {index}:"
    )
    assert (
        "Prompt and response combined length exceeds maximum combined length of 70000 tokens"
        in response["log"]["error"]["message"]
    )
    assert response["log"]["error"]["retryable"] is False


def test_prompt_too_long_exception_single_prompt(tlm: TLM):
    """Tests that bad request error is raised when prompt is too long when calling tlm.prompt with a single prompt."""
    with pytest.raises(TlmBadRequest) as exc_info:
        tlm.prompt(
            "a" * (MAX_PROMPT_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN,
        )

    assert exc_info.value.message.startswith("Prompt length exceeds")
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_prompt_too_long_exception_batch_prompt(tlm: TLM, num_prompts: int):
    """Tests that bad request error is raised when prompt is too long when calling tlm.prompt with a batch of prompts.

    Error message should indicate which the batch index for which the prompt is too long.
    """
    # create batch of prompts with one prompt that is too long
    prompts = [test_prompt] * num_prompts
    prompt_too_long_index = np.random.randint(0, num_prompts)
    prompts[prompt_too_long_index] = "a" * (MAX_PROMPT_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN

    with pytest.raises(TlmBadRequest) as exc_info:
        tlm.prompt(prompts)

    assert exc_info.value.message.startswith(
        f"Error executing query at index {prompt_too_long_index}:"
    )
    assert "Prompt length exceeds" in exc_info.value.message
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_prompt_too_long_exception_try_prompt(tlm: TLM, num_prompts: int):
    """Tests that None is returned when prompt is too long when calling tlm.try_prompt with a batch of prompts."""
    # create batch of prompts with one prompt that is too long
    prompts = [test_prompt] * num_prompts
    prompt_too_long_index = np.random.randint(0, num_prompts)
    prompts[prompt_too_long_index] = "a" * (MAX_PROMPT_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN

    tlm_responses = tlm.try_prompt(
        prompts,
    )

    assert_prompt_too_long_error(tlm_responses[prompt_too_long_index], prompt_too_long_index)


def test_response_too_long_exception_single_score(tlm: TLM):
    """Tests that bad request error is raised when response is too long when calling tlm.get_trustworthiness_score with a single prompt."""
    with pytest.raises(TlmBadRequest) as exc_info:
        tlm.get_trustworthiness_score(
            "a",
            "a" * (MAX_RESPONSE_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN,
        )

    assert exc_info.value.message.startswith("Response length exceeds")
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_response_too_long_exception_batch_score(tlm: TLM, num_prompts: int):
    """Tests that bad request error is raised when prompt is too long when calling tlm.get_trustworthiness_score with a batch of prompts.

    Error message should indicate which the batch index for which the prompt is too long.
    """
    # create batch of prompts with one prompt that is too long
    prompts = [test_prompt] * num_prompts
    responses = [TEST_RESPONSE] * num_prompts
    response_too_long_index = np.random.randint(0, num_prompts)
    responses[response_too_long_index] = (
        "a" * (MAX_RESPONSE_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN
    )

    with pytest.raises(TlmBadRequest) as exc_info:
        tlm.get_trustworthiness_score(
            prompts,
            responses,
        )

    assert exc_info.value.message.startswith(
        f"Error executing query at index {response_too_long_index}:"
    )
    assert "Response length exceeds" in exc_info.value.message
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_response_too_long_exception_try_score(tlm: TLM, num_prompts: int):
    """Tests that None is returned when prompt is too long when calling tlm.try_get_trustworthiness_score with a batch of prompts."""
    # create batch of prompts with one prompt that is too long
    prompts = [test_prompt] * num_prompts
    responses = [TEST_RESPONSE] * num_prompts
    response_too_long_index = np.random.randint(0, num_prompts)
    responses[response_too_long_index] = (
        "a" * (MAX_RESPONSE_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN
    )

    tlm_responses = tlm.try_get_trustworthiness_score(
        prompts,
        responses,
    )

    assert_response_too_long_error_score(
        tlm_responses[response_too_long_index], response_too_long_index
    )


def test_prompt_too_long_exception_single_score(tlm: TLM):
    """Tests that bad request error is raised when prompt is too long when calling tlm.get_trustworthiness_score with a single prompt."""
    with pytest.raises(TlmBadRequest) as exc_info:
        tlm.get_trustworthiness_score(
            "a" * (MAX_PROMPT_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN,
            "a",
        )

    assert exc_info.value.message.startswith("Prompt length exceeds")
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_prompt_too_long_exception_batch_score(tlm: TLM, num_prompts: int):
    """Tests that bad request error is raised when prompt is too long when calling tlm.get_trustworthiness_score with a batch of prompts.

    Error message should indicate which the batch index for which the prompt is too long.
    """
    # create batch of prompts with one prompt that is too long
    prompts = [test_prompt] * num_prompts
    responses = [TEST_RESPONSE] * num_prompts
    prompt_too_long_index = np.random.randint(0, num_prompts)
    prompts[prompt_too_long_index] = "a" * (MAX_PROMPT_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN

    with pytest.raises(TlmBadRequest) as exc_info:
        tlm.get_trustworthiness_score(
            prompts,
            responses,
        )

    assert exc_info.value.message.startswith(
        f"Error executing query at index {prompt_too_long_index}:"
    )
    assert "Prompt length exceeds" in exc_info.value.message
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_prompt_too_long_exception_try_score(tlm: TLM, num_prompts: int):
    """Tests that None is returned when prompt is too long when calling tlm.try_get_trustworthiness_score with a batch of prompts."""
    # create batch of prompts with one prompt that is too long
    prompts = [test_prompt] * num_prompts
    responses = [TEST_RESPONSE] * num_prompts
    prompt_too_long_index = np.random.randint(0, num_prompts)
    prompts[prompt_too_long_index] = "a" * (MAX_PROMPT_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN

    responses = tlm.try_get_trustworthiness_score(
        prompts,
        responses,
    )

    assert_prompt_too_long_error_score(responses[prompt_too_long_index], prompt_too_long_index)


def test_combined_too_long_exception_single_score(tlm: TLM):
    """Tests that bad request error is raised when prompt + response combined length is too long when calling tlm.get_trustworthiness_score with a single prompt."""
    max_prompt_length = MAX_COMBINED_LENGTH_TOKENS - MAX_RESPONSE_LENGTH_TOKENS + 1

    with pytest.raises(TlmBadRequest) as exc_info:
        tlm.get_trustworthiness_score(
            "a" * max_prompt_length * CHARACTERS_PER_TOKEN,
            "a" * MAX_RESPONSE_LENGTH_TOKENS * CHARACTERS_PER_TOKEN,
        )

    assert exc_info.value.message.startswith("Prompt and response combined length exceeds")
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_prompt_and_response_combined_too_long_exception_batch_score(tlm: TLM, num_prompts: int):
    """Tests that bad request error is raised when prompt + response combined length is too long when calling tlm.get_trustworthiness_score with a batch of prompts.

    Error message should indicate which the batch index for which the prompt is too long.
    """
    # create batch of prompts with one prompt that is too long
    prompts = [test_prompt] * num_prompts
    responses = [TEST_RESPONSE] * num_prompts
    combined_too_long_index = np.random.randint(0, num_prompts)

    max_prompt_length = MAX_COMBINED_LENGTH_TOKENS - MAX_RESPONSE_LENGTH_TOKENS + 1
    prompts[combined_too_long_index] = "a" * max_prompt_length * CHARACTERS_PER_TOKEN
    responses[combined_too_long_index] = "a" * MAX_RESPONSE_LENGTH_TOKENS * CHARACTERS_PER_TOKEN

    tlm_responses = tlm.try_get_trustworthiness_score(
        prompts,
        responses,
    )

    assert_prompt_and_response_combined_too_long_error_score(
        tlm_responses[combined_too_long_index], combined_too_long_index
    )


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_prompt_and_response_combined_too_long_exception_try_score(tlm: TLM, num_prompts: int):
    """Tests that appropriate error is returned when prompt + response is too long when calling tlm.try_get_trustworthiness_score with a batch of prompts."""
    # create batch of prompts with one prompt that is too long
    prompts = [test_prompt] * num_prompts
    responses = [TEST_RESPONSE] * num_prompts
    combined_too_long_index = np.random.randint(0, num_prompts)
    max_prompt_length = MAX_COMBINED_LENGTH_TOKENS - MAX_RESPONSE_LENGTH_TOKENS + 1
    prompts[combined_too_long_index] = "a" * max_prompt_length * CHARACTERS_PER_TOKEN
    responses[combined_too_long_index] = "a" * MAX_RESPONSE_LENGTH_TOKENS * CHARACTERS_PER_TOKEN

    responses = tlm.try_get_trustworthiness_score(
        prompts,
        responses,
    )

    assert_prompt_and_response_combined_too_long_error_score(
        responses[combined_too_long_index], combined_too_long_index
    )


def test_invalid_option_passed(studio: Studio):
    """Tests that validation error is thrown when an invalid option is passed to the TLM."""
    invalid_option = "invalid_option"

    with pytest.raises(
        ValidationError, match=f"^Invalid keys in options dictionary: {{'{invalid_option}'}}.*"
    ):
        studio.TLM(options={invalid_option: "invalid_value"})


def test_max_tokens_invalid_option_passed(studio: Studio):
    """Tests that validation error is thrown when an invalid max_tokens option value is passed to the TLM."""
    option = "max_tokens"
    option_value = -1

    with pytest.raises(ValidationError, match=f"Invalid value {option_value}, max_tokens.*"):
        studio.TLM(options={option: option_value})
