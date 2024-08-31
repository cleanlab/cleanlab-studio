import numpy as np
import pytest

from cleanlab_studio.errors import TlmBadRequest, ValidationError
from cleanlab_studio.studio.studio import Studio
from cleanlab_studio.studio.trustworthy_language_model import TLM

np.random.seed(0)


MAX_PROMPT_LENGTH_TOKENS: int = 70_000
MAX_RESPONSE_LENGTH_TOKENS: int = 15_000
MAX_COMBINED_LENGTH_TOKENS: int = 70_000

CHARACTERS_PER_TOKEN: int = 4


def test_prompt_too_long_exception_single_prompt(tlm: TLM):
    """Tests that bad request error is raised when prompt is too long when calling tlm.prompt with a single prompt."""
    with pytest.raises(TlmBadRequest, match="^Prompt length exceeds.*"):
        tlm.prompt(
            "a" * (MAX_PROMPT_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN,
        )


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_prompt_too_long_exception_batch_prompt(tlm: TLM, num_prompts: int):
    """Tests that bad request error is raised when prompt is too long when calling tlm.prompt with a batch of prompts.

    Error message should indicate which the batch index for which the prompt is too long.
    """
    # create batch of prompts with one prompt that is too long
    prompts = ["What is the capital of France?"] * num_prompts
    prompt_too_long_index = np.random.randint(0, num_prompts)
    prompts[prompt_too_long_index] = "a" * (MAX_PROMPT_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN

    with pytest.raises(
        TlmBadRequest,
        match=f"^Error executing query at index {prompt_too_long_index}:\nPrompt length exceeds.*",
    ):
        tlm.prompt(
            prompts,
        )


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_prompt_too_long_exception_try_prompt(tlm: TLM, num_prompts: int):
    """Tests that None is returned when prompt is too long when calling tlm.try_prompt with a batch of prompts."""
    # create batch of prompts with one prompt that is too long
    prompts = ["What is the capital of France?"] * num_prompts
    prompt_too_long_index = np.random.randint(0, num_prompts)
    prompts[prompt_too_long_index] = "a" * (MAX_PROMPT_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN

    tlm_responses = tlm.try_prompt(
        prompts,
    )

    # assert -- None is returned at correct index
    assert tlm_responses[prompt_too_long_index] is None


def test_response_too_long_exception_single_score(tlm: TLM):
    """Tests that bad request error is raised when response is too long when calling tlm.get_trustworthiness_score with a single prompt."""
    with pytest.raises(TlmBadRequest, match="^Response length exceeds.*"):
        tlm.get_trustworthiness_score(
            "a",
            "a" * (MAX_RESPONSE_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN,
        )


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_response_too_long_exception_batch_score(tlm: TLM, num_prompts: int):
    """Tests that bad request error is raised when prompt is too long when calling tlm.get_trustworthiness_score with a batch of prompts.

    Error message should indicate which the batch index for which the prompt is too long.
    """
    # create batch of prompts with one prompt that is too long
    prompts = ["What is the capital of France?"] * num_prompts
    responses = ["Paris"] * num_prompts
    response_too_long_index = np.random.randint(0, num_prompts)
    responses[response_too_long_index] = (
        "a" * (MAX_RESPONSE_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN
    )

    with pytest.raises(
        TlmBadRequest,
        match=f"^Error executing query at index {response_too_long_index}:\nResponse length exceeds.*",
    ):
        tlm.get_trustworthiness_score(
            prompts,
            responses,
        )


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_response_too_long_exception_try_score(tlm: TLM, num_prompts: int):
    """Tests that None is returned when prompt is too long when calling tlm.try_get_trustworthiness_score with a batch of prompts."""
    # create batch of prompts with one prompt that is too long
    prompts = ["What is the capital of France?"] * num_prompts
    responses = ["Paris"] * num_prompts
    response_too_long_index = np.random.randint(0, num_prompts)
    responses[response_too_long_index] = (
        "a" * (MAX_RESPONSE_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN
    )

    tlm_responses = tlm.try_get_trustworthiness_score(
        prompts,
        responses,
    )

    # assert -- None is returned at correct index
    assert tlm_responses[response_too_long_index] is None


def test_prompt_too_long_exception_single_score(tlm: TLM):
    """Tests that bad request error is raised when prompt is too long when calling tlm.get_trustworthiness_score with a single prompt."""
    with pytest.raises(TlmBadRequest, match="^Prompt length exceeds.*"):
        tlm.get_trustworthiness_score(
            "a" * (MAX_PROMPT_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN,
            "a",
        )


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_prompt_too_long_exception_batch_score(tlm: TLM, num_prompts: int):
    """Tests that bad request error is raised when prompt is too long when calling tlm.get_trustworthiness_score with a batch of prompts.

    Error message should indicate which the batch index for which the prompt is too long.
    """
    # create batch of prompts with one prompt that is too long
    prompts = ["What is the capital of France?"] * num_prompts
    responses = ["Paris"] * num_prompts
    prompt_too_long_index = np.random.randint(0, num_prompts)
    prompts[prompt_too_long_index] = "a" * (MAX_PROMPT_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN

    with pytest.raises(
        TlmBadRequest,
        match=f"^Error executing query at index {prompt_too_long_index}:\nPrompt length exceeds.*",
    ):
        tlm.get_trustworthiness_score(
            prompts,
            responses,
        )


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_prompt_too_long_exception_try_score(tlm: TLM, num_prompts: int):
    """Tests that None is returned when prompt is too long when calling tlm.try_get_trustworthiness_score with a batch of prompts."""
    # create batch of prompts with one prompt that is too long
    prompts = ["What is the capital of France?"] * num_prompts
    responses = ["Paris"] * num_prompts
    prompt_too_long_index = np.random.randint(0, num_prompts)
    prompts[prompt_too_long_index] = "a" * (MAX_PROMPT_LENGTH_TOKENS + 1) * CHARACTERS_PER_TOKEN

    responses = tlm.try_get_trustworthiness_score(
        prompts,
        responses,
    )

    # assert -- None is returned at correct index
    assert responses[prompt_too_long_index] is None


def test_combined_too_long_exception_single_score(tlm: TLM):
    """Tests that bad request error is raised when prompt + response combined length is too long when calling tlm.get_trustworthiness_score with a single prompt."""
    max_prompt_length = MAX_COMBINED_LENGTH_TOKENS - MAX_RESPONSE_LENGTH_TOKENS + 1

    with pytest.raises(TlmBadRequest, match="^Prompt and response combined length exceeds.*"):
        tlm.get_trustworthiness_score(
            "a" * max_prompt_length * CHARACTERS_PER_TOKEN,
            "a" * MAX_RESPONSE_LENGTH_TOKENS * CHARACTERS_PER_TOKEN,
        )


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_prompt_too_long_exception_batch_score(tlm: TLM, num_prompts: int):
    """Tests that bad request error is raised when prompt + response combined length is too long when calling tlm.get_trustworthiness_score with a batch of prompts.

    Error message should indicate which the batch index for which the prompt is too long.
    """
    # create batch of prompts with one prompt that is too long
    prompts = ["What is the capital of France?"] * num_prompts
    responses = ["Paris"] * num_prompts
    combined_too_long_index = np.random.randint(0, num_prompts)

    max_prompt_length = MAX_COMBINED_LENGTH_TOKENS - MAX_RESPONSE_LENGTH_TOKENS + 1
    prompts[combined_too_long_index] = "a" * max_prompt_length * CHARACTERS_PER_TOKEN
    responses[combined_too_long_index] = "a" * MAX_RESPONSE_LENGTH_TOKENS * CHARACTERS_PER_TOKEN

    with pytest.raises(
        TlmBadRequest,
        match=f"^Error executing query at index {combined_too_long_index}:\nPrompt and response combined length exceeds.*",
    ):
        tlm.get_trustworthiness_score(
            prompts,
            responses,
        )


@pytest.mark.parametrize("num_prompts", [1, 2, 5])
def test_prompt_too_long_exception_try_score(tlm: TLM, num_prompts: int):
    """Tests that None is returned when prompt + response is too long when calling tlm.try_get_trustworthiness_score with a batch of prompts."""
    # create batch of prompts with one prompt that is too long
    prompts = ["What is the capital of France?"] * num_prompts
    responses = ["Paris"] * num_prompts
    combined_too_long_index = np.random.randint(0, num_prompts)
    prompts[combined_too_long_index] = (
        "a" * (MAX_PROMPT_LENGTH_TOKENS // 2 + 1) * CHARACTERS_PER_TOKEN
    )
    responses[combined_too_long_index] = (
        "a" * (MAX_PROMPT_LENGTH_TOKENS // 2 + 1) * CHARACTERS_PER_TOKEN
    )

    responses = tlm.try_get_trustworthiness_score(
        prompts,
        responses,
    )

    # assert -- None is returned at correct index
    assert responses[combined_too_long_index] is None


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
