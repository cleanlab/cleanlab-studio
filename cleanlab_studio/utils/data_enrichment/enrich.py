from typing import Any, List, Tuple, Union, Dict
from functools import lru_cache
from cleanlab_studio.internal.enrichment_utils import (
    get_prompt_outputs,
    get_regex_match_or_replacement,
    get_constrain_outputs_match,
    get_optimized_prompt,
)
from cleanlab_studio.studio.enrichment import EnrichmentOptions


@lru_cache(maxsize=None)
def _get_pandas():
    import pandas as pd

    return pd


@lru_cache(maxsize=None)
def _get_tqdm():
    from tqdm import tqdm

    return tqdm


def run_online(
    data: Union["pd.DataFrame", List[dict]],
    options: EnrichmentOptions,
    new_column_name: str,
    studio: Any,
) -> Dict[str, Any]:
    """
    Enrich data in real-time using the same logic as the run() method, but client-side.

    Args:
        data (Union[pd.DataFrame, List[dict]]): The dataset to enrich.
        options (EnrichmentOptions): Options for enriching the dataset.
        new_column_name (str): The name of the new column to store the results.
        studio (Any): A required parameter for the Studio object.

    Returns:
        Dict[str, Any]: A dictionary containing information about the enrichment job and the enriched dataset.
    """
    pd = _get_pandas()
    tqdm = _get_tqdm()

    # Validate options
    _validate_enrichment_options(options)

    # Ensure data is a DataFrame
    if isinstance(data, list):
        data = pd.DataFrame(data)

    df = data.copy()

    # Extract options
    prompt = options["prompt"]
    regex = options.get("regex")
    constrain_outputs = options.get("constrain_outputs")
    optimize_prompt = options.get("optimize_prompt", True)
    quality_preset = options.get("quality_preset", "medium")

    if optimize_prompt:
        prompt = get_optimized_prompt(prompt, constrain_outputs)

    outputs = get_prompt_outputs(
        studio, prompt, df, quality_preset=quality_preset, **options.get("tlm_options", {})
    )
    column_name_prefix = new_column_name + "_"

    df[f"{column_name_prefix}trustworthiness"] = [
        output["trustworthiness_score"] if output is not None else None for output in outputs
    ]
    df[f"{new_column_name}"] = [
        output["response"] if output is not None else None for output in outputs
    ]
    df[f"{column_name_prefix}log"] = [
        output["response"] if output is not None else None for output in outputs
    ]

    if regex is None and constrain_outputs is None:
        enriched_df = df[[f"{new_column_name}", f"{column_name_prefix}trustworthiness"]]
    else:
        if regex:
            df[f"{new_column_name}"] = df[f"{new_column_name}"].apply(
                lambda x: get_regex_match_or_replacement(x, regex)
            )

        if constrain_outputs:
            df[f"{new_column_name}"] = df[f"{new_column_name}"].apply(
                lambda x: get_constrain_outputs_match(x, constrain_outputs)
            )

        enriched_df = df[
            [
                f"{new_column_name}",
                f"{column_name_prefix}trustworthiness",
                f"{column_name_prefix}log",
            ]
        ]

    # Simulate the response structure of the run() method
    job_info = {
        "job_id": "run_online",
        "status": "SUCCEEDED",
        "num_rows": len(enriched_df),
        "processed_rows": len(enriched_df),
        "average_trustworthiness_score": enriched_df[f"{column_name_prefix}trustworthiness"].mean(),
        "results": enriched_df,
    }

    return job_info


def _validate_enrichment_options(options: EnrichmentOptions) -> None:
    required_keys = ["prompt"]
    for key in required_keys:
        if key not in options or options[key] is None:
            raise ValueError(f"'{key}' is required in the options.")

    # Validate types and values
    if not isinstance(options["prompt"], str):
        raise TypeError("'prompt' must be a string.")

    if "constrain_outputs" in options and options["constrain_outputs"] is not None:
        if not isinstance(options["constrain_outputs"], list):
            raise TypeError("'constrain_outputs' must be a list if provided.")

    if "optimize_prompt" in options and options["optimize_prompt"] is not None:
        if not isinstance(options["optimize_prompt"], bool):
            raise TypeError("'optimize_prompt' must be a boolean if provided.")

    if "quality_preset" in options and options["quality_preset"] is not None:
        if not isinstance(options["quality_preset"], str):
            raise TypeError("'quality_preset' must be a string if provided.")

    if "regex" in options and options["regex"] is not None:
        regex = options["regex"]
        if not isinstance(regex, (str, tuple, list)):
            raise TypeError("'regex' must be a string, tuple, or list of tuples.")
        if isinstance(regex, list) and not all(isinstance(item, tuple) for item in regex):
            raise TypeError("All items in 'regex' list must be tuples.")


def process_regex(
    column_data: Union["pd.Series", List[str]],
    regex: Union[str, Tuple[str, str], List[Tuple[str, str]]],
) -> Union["pd.Series", List[str]]:
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
    pd = _get_pandas()

    if isinstance(column_data, list):
        return [get_regex_match_or_replacement(x, regex) for x in column_data]
    elif isinstance(column_data, pd.Series):
        return column_data.apply(lambda x: get_regex_match_or_replacement(x, regex))
    else:
        raise TypeError("column_data should be a pandas Series or a list of strings.")
