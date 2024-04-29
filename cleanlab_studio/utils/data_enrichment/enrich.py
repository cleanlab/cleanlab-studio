import re
from typing import List, Optional, Tuple, Union
import pandas as pd
from cleanlab_studio.utils.data_enrichment.enrichment_utils import (
    extract_df_subset,
    get_compiled_regex_list,
    get_prompt_outputs,
    get_regex_match,
    get_return_values_match,
)

from cleanlab_studio.studio.studio import Studio


def enrich_data(
    studio: Studio,
    prompt: str,
    data: pd.DataFrame,
    *,
    regex: Optional[Union[str, re.Pattern, List[re.Pattern]]] = None,
    return_values: Optional[List[str]] = None,
    subset_indices: Optional[Union[Tuple[int, int], List[int]]] = (0, 3),
    column_name_prefix: str = "",
    **kwargs,
) -> pd.DataFrame:
    """
    This method takes in a Studio client object, prompt template and a DataFrame and enriches the dataframe with the results of the prompting and associated trustworthiness scores.

    Args:
        prompt: Formatted f-string, that contains both the prompt, and names of columns to embed.
            **Example:** "Is this a numeric value, answer Yes or No only. Value: {column_name}"
        regex: One or more expressions will be passed into re.compile or a list of already compiled regular expressions.
            If a list is proivded, the regexes are applied in order and first succesfull match is returned.
            **Note:** Regex patterns should each specify exactly 1 group that is the match group using parenthesis like so '.*(<desired match group pattern>)'.
            **Example:** `r'.*(Bird|[Rr]abbit).*'` will match any string that is the word 'Bird', 'Rabbit' or 'rabbit' into group 1.
        return_values: List of all possible values for the `metadata` column.
            If specified, every entry in the `metadata` column will exactly match one of these values (for less open-ended data enrichment tasks). If None, the `metadata` column can contain arbitrary values (for more open-ended data enrichment tasks).
            After your regex is applied, there may be additional transformations applied to ensure the returned value is one of these.
        subset_indices: What subset of the supplied data to run this for. Can be either a list of unique indicies or a range. If None, we run on all the data.
        column_name_prefix: Optional prefix appended to all columns names that are returned.

    Returns:
        A DataFrame that now contains additional `metadata` and `trustworthiness` columns related to the prompt. Columns will have `column_name_prefix_` prepended to them if specified.
        `metadata` column = responses to the prompt and other data mutations if `regex` or `return_values` is not specified.
        `trustworthiness` column = trustworthiness of the prompt responses (which ignore the data mutations).
        **Note**: If any data mutations were made to the original response from the prompt, an additional `log` column will be added to the DataFrame that contains the raw output before the mutations were applied.
    """
    subset_data = extract_df_subset(data, subset_indices)
    outputs = get_prompt_outputs(studio, prompt, subset_data, **kwargs)

    if column_name_prefix != "":
        column_name_prefix = column_name_prefix + "_"

    if (
        regex is None and return_values is None
    ):  # we do not need to have a "logs" column as original output is not augmented by regex or return values
        subset_data[f"{column_name_prefix}metadata"] = [output["response"] for output in outputs]
        subset_data[f"{column_name_prefix}trustworthiness"] = [
            output["trustworthiness_score"] for output in outputs
        ]
        return subset_data

    subset_data[f"{column_name_prefix}logs"] = [output["response"] for output in outputs]
    subset_data[f"{column_name_prefix}trustworthiness"] = [
        output["trustworthiness_score"] for output in outputs
    ]

    if regex:
        regex_list = get_compiled_regex_list(regex)
        subset_data[f"{column_name_prefix}metadata"] = subset_data[
            f"{column_name_prefix}logs"
        ].apply(lambda x: get_regex_match(x, regex_list))
    else:
        subset_data[f"{column_name_prefix}metadata"] = subset_data[f"{column_name_prefix}logs"]

    if return_values:
        return_values_pattern = r"(" + "|".join(return_values) + ")"
        subset_data[f"{column_name_prefix}metadata"] = subset_data[
            f"{column_name_prefix}metadata"
        ].apply(lambda x: get_return_values_match(x, return_values_pattern))

    return subset_data
