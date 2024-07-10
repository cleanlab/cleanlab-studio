from difflib import SequenceMatcher
import re
from typing import Any, List, Optional, Tuple, Union
import warnings

import pandas as pd

from cleanlab_studio.errors import ValidationError
from cleanlab_studio.studio.studio import Studio
from cleanlab_studio.studio.trustworthy_language_model import TLMResponse

Replacement = Tuple[str, str]


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


def get_regex_match_or_replacement(
    response: str, regex: Union[str, Replacement, List[Replacement]]
) -> Optional[str]:
    """Performs regex matching (if input is a string) or replacements (if input is a tuple) to the response
    given the matching patterns and replacement strings.
    """
    # if the regex is a string, do matching
    if isinstance(regex, str):
        compiled_pattern = re.compile(regex)
        pattern_match = compiled_pattern.findall(response)
        if len(pattern_match) > 0:
            return str(pattern_match[0])
        return None

    # if the regex is a tuple (or list of tuples), do replacement
    if isinstance(regex, tuple):
        regex_list = [regex]
    elif isinstance(regex, list):
        regex_list = regex
    else:
        raise ValidationError(
            "Passed in regex has to be either a string (pattern to match), a tuple (pair of patterns to match and replace), "
            "or a list of tuples (pairs of patterns to match and replace)."
        )

    for replacement_pair in regex_list:
        if isinstance(replacement_pair, str):
            raise ValidationError("Only a single regex pattern can be used for matching.")

        if not isinstance(replacement_pair, tuple) or len(replacement_pair) != 2:
            raise ValidationError(
                "Every item of the regex list must be a tuple that contains 2 strings: "
                "(the regex pattern to match, the string to replace the matched pattern with)"
            )

        compiled_pattern = re.compile(replacement_pair[0])
        replacement = replacement_pair[1]
        response = compiled_pattern.sub(replacement, response)

    return response


def get_optimized_prompt(prompt: str, constrain_outputs: Optional[List[str]] = None) -> str:
    """Optimize the prompt by ammending original.
    Adds a pre-prompt message if constrain_outputs are provided. This will help the LLM understand it's response must exactly match one of the return values.
    """

    if constrain_outputs is not None:
        string_constrain_outputs = ", ".join(constrain_outputs)
        pre_prompt = f"Your answer must exactly match one of the following values: [{string_constrain_outputs}].\n"
        optimal_prompt = f"{pre_prompt}{prompt}"
    else:
        optimal_prompt = prompt
    return optimal_prompt


def get_constrain_outputs_match(
    response: str,
    constrain_outputs: List[str],
    constrain_outputs_pattern: Optional[str] = None,
    disable_warnings: bool = True,
) -> str:
    """Extracts the provided output values from the response using regex patterns. Return first extracted value if multiple exist.
    If no value out of the possible `constrain_outputs` is directly mentioned in the response, the return value with greatest string similarity to the response is returned (along with a warning).
    If there are no close matches between the LLM response and any of the possible `constrain_outputs`, then the last entry of the `constrain_outputs` list is returned.

    Params
    ------
    response: Response from the LLM
    constrain_outputs: List of expected output values, the last value of this list should be considered the default/baseline value (eg. “other”),
      that value will be returned if there are no close matches.
    constrain_outputs_pattern: Pre-compiled pattern of all output values. If not specified, pattern is created.
    disable_warnings: If True, print warnings are disabled
    """

    response_str = str(response)

    if constrain_outputs_pattern is None:
        escaped_constrain_outputs = [re.escape(output) for output in constrain_outputs]
        constrain_outputs_pattern = "(" + "|".join(escaped_constrain_outputs) + ")"

    # Parse category if LLM response is properly formatted
    exact_matches = re.findall(constrain_outputs_pattern, response_str, re.IGNORECASE)
    if len(exact_matches) > 0:
        return str(exact_matches[0])

    # If there are no exact matches to a specific category, return the closest category based on string similarity.
    best_match = max(
        constrain_outputs, key=lambda x: SequenceMatcher(None, response_str, x).ratio()
    )
    similarity_score = SequenceMatcher(None, response_str, best_match).ratio()

    if similarity_score < 0.5:
        warning_message = (
            f"None of the constrain_outputs remotely match raw LLM output: {response_str}.\n"
            + "Returning the last entry in the constrain outputs list."
        )
        best_match = constrain_outputs[-1]

    else:
        warning_message = f"None of the constrain_outputs match raw LLM output: {response_str}"

    if not disable_warnings:
        warnings.warn(warning_message)

    return best_match
