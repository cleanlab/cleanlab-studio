from typing import Union, Sequence


def validate_tlm_prompt(prompt: Union[str, Sequence[str]]) -> None:
    if isinstance(prompt, str):
        return

    elif isinstance(prompt, Sequence):
        if any(not isinstance(p, str) for p in prompt):
            raise ValueError("All prompts must be strings.")

    else:
        raise ValueError("prompt must be a string or list of strings.")


def validate_tlm_prompt_response(
    prompt: Union[str, Sequence[str]], response: Union[str, Sequence[str]]
) -> None:
    if isinstance(prompt, str):
        if not isinstance(response, str):
            raise ValueError("responses must be a single string for single prompt.")

    elif isinstance(prompt, Sequence):
        if not isinstance(response, Sequence):
            raise ValueError(
                "responses must be a list or iterable of strings when prompt is a list or iterable."
            )
        if len(prompt) != len(response):
            raise ValueError("Length of prompt and response must match.")

        if any(not isinstance(p, str) for p in prompt):
            raise ValueError("All prompts must be strings.")
        if any(not isinstance(r, str) for r in response):
            raise ValueError("All responses must be strings.")

    else:
        raise ValueError("prompt must be a string or list/iterable of strings.")
