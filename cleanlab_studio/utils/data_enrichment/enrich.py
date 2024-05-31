import re
from typing import Any, List, Optional, Tuple, Union
import pandas as pd
from cleanlab_studio.internal.enrichment_utils import (
    extract_df_subset,
    get_prompt_outputs,
    get_regex_replacement,
    get_constrain_outputs_match,
    get_optimized_prompt,
    Replacement,
)

from cleanlab_studio.studio.studio import Studio


def enrich_data(
    studio: Studio,
    prompt: str,
    data: pd.DataFrame,
    *,
    replacements: Optional[Union[Replacement, List[Replacement]]] = None,
    constrain_outputs: Optional[List[str]] = None,
    optimize_prompt: bool = True,
    subset_indices: Optional[Union[Tuple[int, int], List[int]]] = (0, 3),
    new_column_name: str = "metadata",
    disable_warnings: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Generate a column of arbitrary metadata for your DataFrame, reliably at scale with Generative AI.
    The metadata is separately generated for each row of your DataFrame, based on a prompt that specifies what information you need and what existing columns' data it should be derived from.
    Each row of generated metadata is accompanied by a trustworthiness score, which helps you discover which metadata is most/least reliable.
    You can optionally apply regular expressions to further reformat your metadata beyond raw LLM outputs, or specify that each row of the metadata must be constrained to a particular set of values.

    Args:
        studio: Cleanlab Studio client object, which you must instantiate before calling this method.
        prompt: Formatted f-string, that contains both the prompt, and names of columns to embed.
            **Example:** "Is this a numeric value, answer Yes or No only. Value: {column_name}"
        replacements: A tuple or list of tuples each containing a pair of items:
            - a regex pattern to match (str or re.Pattern)
            - a string to replace the matched pattern with (str)
            These tuples specify the desired patterns to match and replace from the raw LLM response,
            it is useful in settings where you are unable to prompt the LLM to generate valid outputs 100% of the time,
            but can easily transform the raw LLM outputs to be valid through regular expressions that extract and replace parts of the raw output string.
            If a list of tuples is passed in, the replacements are applied in the order they appear in the list.

            **Example 1:** ``replacements = (r'\b(?!(True|False)\b)\w+\b', '')`` will replace all words not True or False with an empty string.
            **Example 2:** ``replacements = (re.compile(r' Explanation:.*' re.IGNORECASE), '') will remove everything after and including the words "Explanation:".
            For instance, the response "True. Explanation: 3+4=7, and 7 is an odd number." would return "True" after the regex replacement.
        constrain_outputs: List of all possible values for the `metadata` column.
            If specified, every entry in the `metadata` column will exactly match one of these values (for less open-ended data enrichment tasks). If None, the `metadata` column can contain arbitrary values (for more open-ended data enrichment tasks).
            After your regex is applied, there may be additional transformations applied to ensure the returned value is one of these.
            If `optimize_prompt` is True, the prompt will be automatically adjusted to include a statement that the response must match one of the `constrain_outputs`.
        optimize_prompt: When False, your provided prompt will not be modified in any way. When True, your provided prompt may be automatically adjusted in an effort to produce better results.
            For instance, if the constrain_outputs are constrained, we may automatically append the following statement to your prompt: "Your answer must exactly match one of the following values: `constrain_outputs`."
        subset_indices: What subset of the supplied data rows to generate metadata for. If None, we run on all of the data.
            This can be either a list of unique indices or a range. These indices are passed into pandas ``.iloc`` method, so should be integers based on row order as opposed to row-index labels pointing to `df.index`.
            We advise against collecting results for all of your data at first. First collect results for a smaller data subset, and use this subset to experiment with different values of the `prompt` or `regex` arguments. Only once the results look good for your subset should you run on the full dataset.
        new_column_name: Optional name for the returned metadata column. Name acts as a prefix appended to all additional columns that are returned.
        disable_warnings: When True, warnings are disabled.

    Returns:
        A DataFrame that contains `metadata` and `trustworthiness` columns related to the prompt in order of original data. Some columns names will have `new_column_name` prepended to them.
        `metadata` column = responses to the prompt and other data mutations if `replacements` or `constrain_outputs` is not specified.
        `trustworthiness` column = trustworthiness of the prompt responses (which ignore the data mutations).
        **Note**: If you specified the `replacements` or `constrain_outputs` arguments, some additional transformations may be applied to raw LLM outputs to produce the returned values. In these cases, an additional `log` column will be added to the returned DataFrame that records the raw LLM outputs (feel free to disregard these).
    """
    if subset_indices:
        df = extract_df_subset(data, subset_indices)
    else:
        df = data.copy()

    if optimize_prompt:
        prompt = get_optimized_prompt(prompt, constrain_outputs)

    outputs = get_prompt_outputs(studio, prompt, df, **kwargs)
    column_name_prefix = new_column_name + "_"

    df[f"{column_name_prefix}trustworthiness"] = [
        output["trustworthiness_score"] if output is not None else None for output in outputs
    ]
    df[f"{new_column_name}"] = [
        output["response"] if output is not None else None for output in outputs
    ]

    if (
        replacements is None and constrain_outputs is None
    ):  # we do not need to have a "log" column as original output is not augmented by regex or return values
        return df[[f"{new_column_name}", f"{column_name_prefix}trustworthiness"]]

    df[f"{column_name_prefix}log"] = [
        output["response"] if output is not None else None for output in outputs
    ]

    if replacements:
        df[f"{new_column_name}"] = df[f"{new_column_name}"].apply(
            lambda x: get_regex_replacement(x, replacements)
        )

    if constrain_outputs:
        constrain_outputs_pattern = r"(" + "|".join(constrain_outputs) + ")"
        df[f"{new_column_name}"] = df[f"{new_column_name}"].apply(
            lambda x: get_constrain_outputs_match(
                x, constrain_outputs, constrain_outputs_pattern, disable_warnings
            )
        )

    return df[
        [
            f"{new_column_name}",
            f"{column_name_prefix}trustworthiness",
            f"{column_name_prefix}log",
        ]
    ]


def get_regex_replacements(
    column_data: Union[pd.Series, List[str]],
    replacements: Union[Replacement, List[Replacement]],
) -> Union[pd.Series, List[str]]:
    """
    Performs regex replacements to the given string according to the given matching patterns and replacement strings.

    Use this function for: tuning regex replacements to obtain the best outputs from the raw LLM responses for your dataset obtained via ``enrich_data()``, without having to re-run the LLM.
    If a list of tuples is passed in, the replacements are applied in the order they appear in the list.

    **Example 1:** ``replacements = (r'\b(?!(True|False)\b)\w+\b', '')`` will replace all words not True or False with an empty string.
    **Example 2:** ``replacements = (re.compile(r' Explanation:.*' re.IGNORECASE), '') will remove everything after and including the words "Explanation:".
    For instance, the response "True. Explanation: 3+4=7, and 7 is an odd number." would return "True" after the regex replacement.

    Args:
        column_data: A pandas Series or list of strings, where you want to apply a regex to extract matches from each element. This could be the `metadata` column output by ``enrich_data()``.
        replacements: A tuple or list of tuples each containing a pair of items:
            - a regex pattern to match (str or re.Pattern)
            - a string to replace the matched pattern with (str)

    Returns:
        Extracted matches to the provided regular expression from each element of the data column (specifically, the first match is returned).
    """
    if isinstance(column_data, list):
        return [get_regex_replacement(x, replacements) for x in column_data]
    elif isinstance(column_data, pd.Series):
        return column_data.apply(lambda x: get_regex_replacement(x, replacements))
    else:
        raise TypeError("column_data should be a pandas Series or a list of strings.")
