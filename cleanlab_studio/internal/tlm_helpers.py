from typing import Union, Sequence, Any
from cleanlab_studio.errors import ValidationError
from cleanlab_studio.internal.constants import (
    _VALID_TLM_MODELS,
    TLM_MAX_TOKEN_RANGE,
    TLM_NUM_CANDIDATE_RESPONSES_RANGE,
    TLM_NUM_CONSISTENCY_SAMPLES_RANGE,
)


def validate_tlm_prompt(prompt: Union[str, Sequence[str]]) -> None:
    if isinstance(prompt, str):
        return

    elif isinstance(prompt, Sequence):
        if any(not isinstance(p, str) for p in prompt):
            raise ValidationError(
                "Some items in prompt are of invalid types, all items in the prompt list must be of type str."
            )

    else:
        raise ValidationError(
            f"Invalid type {type(prompt)}, prompt must either be strings or list/iterable of strings."
        )


def validate_tlm_prompt_response(
    prompt: Union[str, Sequence[str]], response: Union[str, Sequence[str]]
) -> None:
    if isinstance(prompt, str):
        if not isinstance(response, str):
            raise ValidationError(
                "response type must match prompt type. "
                f"prompt was provided as str but response is of type {type(response)}"
            )

    elif isinstance(prompt, Sequence):
        if not isinstance(response, Sequence):
            raise ValidationError(
                "response type must match prompt type. "
                f"prompt was provided as type {type(prompt)} but response is of type {type(response)}"
            )

        if len(prompt) != len(response):
            raise ValidationError("Length of the prompt and response lists must match.")

        if any(not isinstance(p, str) for p in prompt):
            raise ValidationError(
                "Some items in prompt are of invalid types, all items in the prompt list must be of type str."
            )
        if any(not isinstance(r, str) for r in response):
            raise ValidationError(
                "Some items in response are of invalid types, all items in the response list must be of type str."
            )

    else:
        raise ValidationError(
            f"Invalid type {type(prompt)}, prompt must either be strings or list/iterable of strings."
        )


def validate_tlm_options(options: Any) -> None:
    from cleanlab_studio.studio.trustworthy_language_model import TLMOptions

    if not isinstance(options, dict):
        raise ValidationError(
            "options must be a TLMOptions object.\n"
            "See: https://help.cleanlab.ai/reference/python/trustworthy_language_model/#class-tlmoptions"
        )

    invalid_keys = set(options.keys()) - set(TLMOptions.__annotations__.keys())
    if invalid_keys:
        raise ValidationError(
            f"Invalid keys in options dictionary: {invalid_keys}.\n"
            "See https://help.cleanlab.ai/reference/python/trustworthy_language_model/#class-tlmoptions for valid options"
        )

    for option, val in options.items():
        if option == "max_tokens":
            if not isinstance(val, int):
                raise ValidationError(f"Invalid type {type(val)}, max_tokens must be an integer")

            if val < TLM_MAX_TOKEN_RANGE[0] or val > TLM_MAX_TOKEN_RANGE[1]:
                raise ValidationError(
                    f"Invalid value {val}, max_tokens must be in the range {TLM_MAX_TOKEN_RANGE}"
                )

        elif option == "model":
            if val not in _VALID_TLM_MODELS:
                raise ValidationError(
                    f"{val} is not a supported model, valid models include: {_VALID_TLM_MODELS}"
                )

        elif option == "num_candidate_responses":
            if not isinstance(val, int):
                raise ValidationError(
                    f"Invalid type {type(val)}, num_candidate_responses must be an integer"
                )

            if (
                val < TLM_NUM_CANDIDATE_RESPONSES_RANGE[0]
                or val > TLM_NUM_CANDIDATE_RESPONSES_RANGE[1]
            ):
                raise ValidationError(
                    f"Invalid value {val}, num_candidate_responses must be in the range {TLM_NUM_CANDIDATE_RESPONSES_RANGE}"
                )

        elif option == "num_consistency_samples":
            if not isinstance(val, int):
                raise ValidationError(
                    f"Invalid type {type(val)}, num_consistency_samples must be an integer"
                )

            if (
                val < TLM_NUM_CONSISTENCY_SAMPLES_RANGE[0]
                or val > TLM_NUM_CONSISTENCY_SAMPLES_RANGE[1]
            ):
                raise ValidationError(
                    f"Invalid value {val}, num_consistency_samples must be in the range {TLM_NUM_CONSISTENCY_SAMPLES_RANGE}"
                )

        elif option == "use_self_reflection":
            if not isinstance(val, bool):
                raise ValidationError(
                    f"Invalid type {type(val)}, use_self_reflection must be a boolean"
                )
