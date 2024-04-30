import re
from typing import List, Optional, Tuple, Union
import pandas as pd
from cleanlab_studio.utils.data_enrichment.enrichment_utils import (
    extract_df_subset,
    get_compiled_regex_list,
    get_prompt_outputs,
    get_regex_match,
    get_return_values_match,
    get_optimized_prompt,
)

from cleanlab_studio.studio.studio import Studio


def enrich_data(
    studio: Studio,
    prompt: str,
    data: pd.DataFrame,
    *,
    regex: Optional[Union[str, re.Pattern[str], List[re.Pattern[str]]]] = None,
    return_values: Optional[List[str]] = None,
    optimize_prompt: bool = True,
    subset_indices: Optional[Union[Tuple[int, int], List[int]]] = (0, 3),
    metadata_column_name: str = "metadata",
    disable_warnings: bool = False,
    **kwargs,
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
        regex: One or more expressions will be passed into ``re.compile()`` or a list of already compiled regular expressions.
            The regex will be applied to the raw LLM outputs from your prompt, enabling additional control over the final column values returned.
            If a list is provided, the regexes are applied in order and first successful match is returned.
            This regex argument is useful in settings where you are unable to prompt the LLM to generate valid outputs 100% of the time, but can easily transform the raw LLM outputs to be valid through regular expressions that extract or replace parts of the raw output string.

            **Note:** Regex patterns should each specify exactly 1 group that is the match group using parenthesis like so '.*(<desired match group pattern>)'.
            **Example:** `r'.*(Bird|[Rr]abbit).*'` will match any string that is the word 'Bird', 'Rabbit' or 'rabbit' into group 1.
        return_values: List of all possible values for the `metadata` column.
            If specified, every entry in the `metadata` column will exactly match one of these values (for less open-ended data enrichment tasks). If None, the `metadata` column can contain arbitrary values (for more open-ended data enrichment tasks).
            After your regex is applied, there may be additional transformations applied to ensure the returned value is one of these.
            If `optimize_prompt` is True, the prompt will be automatically adjusted to include a statement that the response must match one of the `return_values`.
        optimize_prompt: When False, your provided prompt will not be modified in any way. When True, your provided prompt may be automatically adjusted in an effort to produce better results.
            For instance, if the return_values are constrained, we may automatically append the following statement to your prompt: "Your answer must exactly match one of the following values: `return_values`."
        subset_indices: What subset of the supplied data rows to generate metadata for. If None, we run on all of the data.
            This can be either a list of unique indices or a range. These indices are passed into pandas ``.iloc`` method, so should be integers based on row order as opposed to row-index labels pointing to `df.index`.
            We advise against collecting results for all of your data at first. First collect results for a smaller data subset, and use this subset to experiment with different values of the `prompt` or `regex` arguments. Only once the results look good for your subset should you run on the full dataset.
        metadata_column_name: Optional name for the returned metadata column. Nmae acts as a prefix appended to all additional columns that are returned.
        disable_warnings: When True, warnings are disabled.

    Returns:
        A DataFrame that contains `metadata` and `trustworthiness` columns related to the prompt in order of original data. Columns will have `column_name_prefix_` prepended to them if specified.
        `metadata` column = responses to the prompt and other data mutations if `regex` or `return_values` is not specified.
        `trustworthiness` column = trustworthiness of the prompt responses (which ignore the data mutations).
        **Note**: If you specified the `regex` or `return_values` arguments, some additional transformations may be applied to raw LLM outputs to produce the returned values. In these cases, an additional `log` column will be added to the returned DataFrame that records the raw LLM outputs.
    """
    if subset_indices:
        df = extract_df_subset(data, subset_indices)
    else:
        df = data.copy()

    if optimize_prompt:
        prompt = get_optimized_prompt(prompt, return_values)

    outputs = get_prompt_outputs(studio, prompt, df, **kwargs)
    column_name_prefix = metadata_column_name + "_"

    if (
        regex is None and return_values is None
    ):  # we do not need to have a "log" column as original output is not augmented by regex or return values
        df[f"{metadata_column_name}"] = [output["response"] for output in outputs]
        df[f"{column_name_prefix}trustworthiness"] = [
            output["trustworthiness_score"] for output in outputs
        ]
        return df[[f"{metadata_column_name}", f"{column_name_prefix}trustworthiness"]]

    df[f"{column_name_prefix}log"] = [output["response"] for output in outputs]
    df[f"{column_name_prefix}trustworthiness"] = [
        output["trustworthiness_score"] for output in outputs
    ]

    if regex:
        regex_list = get_compiled_regex_list(regex)
        df[f"{metadata_column_name}"] = df[f"{column_name_prefix}log"].apply(
            lambda x: get_regex_match(x, regex_list)
        )
    else:
        df[f"{metadata_column_name}"] = df[f"{column_name_prefix}log"]

    if return_values:
        return_values_pattern = r"(" + "|".join(return_values) + ")"
        df[f"{metadata_column_name}"] = df[f"{metadata_column_name}"].apply(
            lambda x: get_return_values_match(
                x, return_values, return_values_pattern, disable_warnings
            )
        )

    return df[
        [
            f"{metadata_column_name}",
            f"{column_name_prefix}trustworthiness",
            f"{column_name_prefix}log",
        ]
    ]


def get_regex_matches(
    column_data: Union[pd.Series, List[str]],
    regex: Union[str, re.Pattern[str], List[re.Pattern[str]]],
) -> Union[pd.Series, List[str]]:
    """
    Extracts the first match from the response using the provided regex patterns. Return first match if multiple exist.
    Note: This function assumes the regex patterns each specify exactly 1 group that is the match group using '(<group>)'.

    Args:
        column_data: A pandas series or list of strings that you want to apply the regex to.
        regex: A single regex pattern or a list of regex patterns to apply to the column_data.

    Returns:
        The first matches of each response using the provided regex patterns.
    """
    regex_list = get_compiled_regex_list(regex)
    if isinstance(column_data, list):
        return [get_regex_match(x, regex_list) for x in column_data]
    elif isinstance(column_data, pd.Series):
        return column_data.apply(lambda x: get_regex_match(x, regex_list))
    else:
        raise TypeError("column_data should be a pandas Series or a list of strings.")
