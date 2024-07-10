from typing import Any, List, Optional, Tuple, Union
import pandas as pd
from cleanlab_studio.internal.enrichment_utils import (
    extract_df_subset,
    get_prompt_outputs,
    get_regex_match_or_replacement,
    get_constrain_outputs_match,
    get_optimized_prompt,
    Replacement,
)

from cleanlab_studio.studio.studio import Studio


def enrich_data(
    studio: Studio,
    data: pd.DataFrame,
    prompt: str,
    *,
    regex: Optional[Union[str, Replacement, List[Replacement]]] = None,
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
        studio (Studio): Cleanlab Studio client object, which you must instantiate before calling this method.
        data (pd.DataFrame): A pandas DataFrame containing your data.
        prompt (str): Formatted f-string, that contains both the prompt, and names of columns to embed.
            **Example:** "Is this a numeric value, answer Yes or No only. Value: {column_name}"
        regex (str | Replacement | List[Replacement], optional): A string, tuple, or list of tuples specifying regular expressions to apply for post-processing the raw LLM outputs.

            If a string value is passed in, a regex match will be performed and the matched pattern will be returned (if the pattern cannot be matched, None will be returned).
            Specifically the provided string will be passed into Python's `re.match()` method.
            Pass in a tuple `(R1, R2)` instead if you wish to perform find and replace operations rather than matching/extraction.
            `R1` should be a string containing the regex pattern to match, and `R2` should be a string to replace matches with.
            Pass in a list of tuples instead if you wish to apply multiple replacements. Replacements will be applied in the order they appear in the list.
            Note that you cannot pass in a list of strings (chaining of multiple regex processing steps is only allowed for replacement operations).

            These tuples specify the desired patterns to match and replace from the raw LLM response,
            This regex processing is useful in settings where you are unable to prompt the LLM to generate valid outputs 100% of the time,
            but can easily transform the raw LLM outputs to be valid through regular expressions that extract and replace parts of the raw output string.
            When this regex is applied, the processed results can be seen ithe ``{new_column_name}`` column, and the raw outpus (before any regex processing)
            will be saved in the ``{new_column_name}_log`` column of the results dataframe.

            **Example 1:** ``regex = '.*The answer is: (Bird|[Rr]abbit).*'`` will extract strings that are the words 'Bird', 'Rabbit' or 'rabbit' after the characters "The answer is: " from the raw response.
            **Example 2:** ``regex = [('True', 'T'), ('False', 'F')]`` will replace the words True and False with T and F.
            **Example 3:** ``regex = (' Explanation:.*', '') will remove everything after and including the words "Explanation:".
            For instance, the response "True. Explanation: 3+4=7, and 7 is an odd number." would return "True." after the regex replacement.
        constrain_outputs (List[str], optional): List of all possible output values for the `metadata` column.
            If specified, every entry in the `metadata` column will exactly match one of these values (for less open-ended data enrichment tasks). If None, the `metadata` column can contain arbitrary values (for more open-ended data enrichment tasks).
            There may be additional transformations applied to ensure the returned value is one of these. If regex is also specified, then these transformations occur after your regex is applied.
            If `optimize_prompt` is True, the prompt will be automatically adjusted to include a statement that the response must match one of the `constrain_outputs`.
            The last value of this list should be considered the baseline value (eg. “other”) that will be returned where there are no close matches between the raw LLM response and any of the classes mentioned,
            that value will be returned if no close matches can be made.
        optimize_prompt (bool, default = True): When False, your provided prompt will not be modified in any way. When True, your provided prompt may be automatically adjusted in an effort to produce better results.
            For instance, if the constrain_outputs are constrained, we may automatically append the following statement to your prompt: "Your answer must exactly match one of the following values: `constrain_outputs`."
        subset_indices (Tuple[int, int] | List[int], optional): What subset of the supplied data rows to generate metadata for. If None, we run on all of the data.
            This can be either a list of unique indices or a range. These indices are passed into pandas ``.iloc`` method, so should be integers based on row order as opposed to row-index labels pointing to `df.index`.
            We advise against collecting results for all of your data at first. First collect results for a smaller data subset, and use this subset to experiment with different values of the `prompt` or `regex` arguments. Only once the results look good for your subset should you run on the full dataset.
        new_column_name (str): Optional name for the returned enriched column. Name acts as a prefix appended to all additional columns that are returned.
        disable_warnings (bool, default = False): When True, warnings are disabled.
        **kwargs: Optional keyword arguments to pass to the underlying TLM object, such as ``quality_preset`` and ``options`` to specify the TLM quality present and TLMOptions respectively.
            For more information on valid TLM arguments, view the TLM documentation here: https://help.cleanlab.ai/reference/python/studio/#method-tlm

    Returns:
        A DataFrame that contains `metadata` and `trustworthiness` columns related to the prompt in order of original data. Some columns names will have `new_column_name` prepended to them.
        `metadata` column = responses to the prompt and other data mutations if `regex` or `constrain_outputs` is not specified.
        `trustworthiness` column = trustworthiness of the prompt responses (which ignore the data mutations).
        **Note**: If you specified the `regex` or `constrain_outputs` arguments, some additional transformations may be applied to raw LLM outputs to produce the returned values. In these cases, an additional `log` column will be added to the returned DataFrame that records the raw LLM outputs (feel free to disregard these).
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
        regex is None and constrain_outputs is None
    ):  # we do not need to have a "log" column as original output is not augmented by regex replacements or contrained outputs
        return df[[f"{new_column_name}", f"{column_name_prefix}trustworthiness"]]

    df[f"{column_name_prefix}log"] = [
        output["response"] if output is not None else None for output in outputs
    ]

    if regex:
        df[f"{new_column_name}"] = df[f"{new_column_name}"].apply(
            lambda x: get_regex_match_or_replacement(x, regex)
        )

    if constrain_outputs:
        df[f"{new_column_name}"] = df[f"{new_column_name}"].apply(
            lambda x: get_constrain_outputs_match(
                x, constrain_outputs, disable_warnings=disable_warnings
            )
        )

    return df[
        [
            f"{new_column_name}",
            f"{column_name_prefix}trustworthiness",
            f"{column_name_prefix}log",
        ]
    ]


def process_regex(
    column_data: Union[pd.Series, List[str]],
    regex: Union[str, Replacement, List[Replacement]],
) -> Union[pd.Series, List[str]]:
    """
    Performs regex matches or replacements to the given string according to the given matching patterns and replacement strings.

    Use this function for: tuning regex replacements to obtain the best outputs from the raw LLM responses for your dataset obtained via ``enrich_data()``, without having to re-run the LLM.

    If a string value is passed in, a regex match will be performed and the matched pattern will be returned (if the pattern cannot be matched, None will be returned).
    Specifically the provided string will be passed into Python's `re.match()` method.
    Pass in a tuple `(R1, R2)` instead if you wish to perform find and replace operations rather than matching/extraction.
    `R1` should be a string containing the regex pattern to match, and `R2` should be a string to replace matches with.
    Pass in a list of tuples instead if you wish to apply multiple replacements. Replacements will be applied in the order they appear in the list.
    Note that you cannot pass in a list of strings (chaining of multiple regex processing steps is only allowed for replacement operations).

    **Example 1:** ``regex = '.*The answer is: (Bird|[Rr]abbit).*'`` will extract strings that are the words 'Bird', 'Rabbit' or 'rabbit' after the characters "The answer is: " from the raw response.
    **Example 2:** ``regex = [('True', 'T'), ('False', 'F')]`` will replace the words True and False with T and F.
    **Example 3:** ``regex = (' Explanation:.*', '') will remove everything after and including the words "Explanation:".
    For instance, the response "True. Explanation: 3+4=7, and 7 is an odd number." would return "True." after the regex replacement.

    Args:
        column_data (pd.Series | List[str]): A pandas Series or list of strings, where you want to apply a regex to extract matches from each element. This could be the `metadata` column output by ``enrich_data()``.
        regex (str | Replacement | List[Replacement]): A string, tuple, or list of tuples specifying regular expressions to apply for post-processing the raw LLM outputs.

    Returns:
        Extracted matches to the provided regular expression from each element of the data column (specifically, the first match is returned).
    """
    if isinstance(column_data, list):
        return [get_regex_match_or_replacement(x, regex) for x in column_data]
    elif isinstance(column_data, pd.Series):
        return column_data.apply(lambda x: get_regex_match_or_replacement(x, regex))
    else:
        raise TypeError("column_data should be a pandas Series or a list of strings.")
