from difflib import SequenceMatcher
import re
from typing import Any, List, Optional, Tuple, Union
import warnings

import pandas as pd

from cleanlab_studio.errors import ValidationError
from cleanlab_studio.studio.studio import Studio
from cleanlab_studio.studio.trustworthy_language_model import TLMResponse

Replacement = Tuple[Union[str, re.Pattern[str]], str]


def get_prompt_outputs(
    studio: Studio, prompt: str, data: pd.DataFrame, **kwargs: Any
) -> List[Optional[TLMResponse]]:
    """Returns the outputs of the prompt for each row in the dataframe."""
    default_tlm_options = {"model": "claude-3-haiku"}
    tlm_options = kwargs.get("options", {})
    kwargs["options"] = {**default_tlm_options, **tlm_options}

    tlm = studio.TLM(**kwargs)
    formatted_prompts = data.apply(lambda x: prompt.format(**x), axis=1).to_list()
    outputs = tlm.try_prompt(formatted_prompts)
    return outputs


def extract_df_subset(
    df: pd.DataFrame, subset_indices: Union[Tuple[int, int], List[int], None, range]
) -> pd.DataFrame:
    """Extract a subset of the dataframe based on the provided indices. If no indices are provided, the entire dataframe is returned. Indices can be range or specific row indices."""
    if subset_indices is None:
        print(
            "Processing your full dataset since `subset_indices` is None. This may take a while. Specify this argument to get faster results for a subset of your dataset."
        )
        return df
    if isinstance(subset_indices, range):
        subset_indices = subset_indices
    if isinstance(subset_indices, tuple):
        subset_indices = range(*subset_indices)
    subset_df = df.iloc[subset_indices].copy()
    return subset_df


def get_compiled_regex(regex: Union[str, re.Pattern[str]]) -> re.Pattern[str]:
    """Compile the regex pattern(s) provided and return the compiled regex pattern."""
    if isinstance(regex, str):
        return re.compile(rf"{regex}")
    elif isinstance(regex, re.Pattern):
        return regex
    else:
        raise ValidationError("Passed in regex can only be type one of: str, re.Pattern.")


def get_regex_replacement(
    response: str, regex_replacements: Union[Replacement, List[Replacement]]
) -> Optional[str]:
    """Performs regex replacements to the given string according to the given matching patterns and replacement strings."""
    if isinstance(regex_replacements, tuple):
        regex_replacements_list = [regex_replacements]
    elif isinstance(regex_replacements, list):
        regex_replacements_list = regex_replacements
    else:
        raise ValidationError("Passed in regex has to be either a tuple or a list of tuples.")

    for replacement_pair in regex_replacements_list:
        if not isinstance(replacement_pair, tuple) or len(replacement_pair) != 2:
            raise ValidationError(
                "Every item of the regex list must be a tuple that contains 2 items. TODO"
            )

        compiled_pattern = get_compiled_regex(replacement_pair[0])
        replacement = replacement_pair[1]
        response = compiled_pattern.sub(replacement, response)

    return response


def get_optimized_prompt(prompt: str, return_values: Optional[List[str]] = None) -> str:
    """Optimize the prompt by ammending original.
    Adds a pre-prompt message if return_values are provided. This will help the LLM understand it's response must exactly match one of the return values.
    """

    if return_values is not None:
        string_return_values = str(return_values).replace("'", "")
        pre_prompt = f"Your answer must exactly match one of the following values: [{string_return_values}].\n"
        optimal_prompt = f"{pre_prompt}{prompt}"
    else:
        optimal_prompt = prompt
    return optimal_prompt


def get_return_values_match(
    response: str,
    return_values: List[str],
    return_values_pattern: Optional[str] = None,
    disable_warnings: bool = True,
) -> str:
    """Extracts the provided return values from the response using regex patterns. Return first extracted value if multiple exist. If no value out of the possible `return_values` is directly mentioned in the response, the return value with greatest string similarity to the response is returned (along with a warning).

    Params
    ------
    response: Response from the LLM
    return_values: List of expected return values
    return_values_pattern: Pre-compiled pattern of all return values. If not specified, pattern is created.
    disable_warnings: If True, print warnings are disabled
    """

    response_str = str(response)

    if return_values_pattern is None:
        return_values_pattern = r"(" + "|".join(return_values) + ")"

    # Parse category if LLM response is properly formatted
    exact_matches = re.findall(return_values_pattern, response_str, re.IGNORECASE)
    if len(exact_matches) > 0:
        return str(exact_matches[0])

    # If there are no exact matches to a specific category, return the closest category based on string similarity.
    closest_match = max(return_values, key=lambda x: SequenceMatcher(None, response_str, x).ratio())
    similarity_score = SequenceMatcher(None, response_str, closest_match).ratio()
    str_warning = "match"
    if similarity_score < 0.5:
        str_warning = "remotely match"
    if not disable_warnings:
        warnings.warn(f"None of the return_values {str_warning} raw LLM output: {response_str}")
    return closest_match
