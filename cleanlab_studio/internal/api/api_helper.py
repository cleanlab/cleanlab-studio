import re
from typing import List, Tuple, Union
import uuid

import pandas as pd
from cleanlab_studio.errors import InvalidUUIDError


def check_uuid_well_formed(uuid_string: str, id_name: str) -> None:
    try:
        uuid.UUID(uuid_string)
    except ValueError:
        raise InvalidUUIDError(
            f"{uuid_string} is not a well-formed {id_name}, please double check and try again."
        )


def extract_df_subset(
    df: pd.DataFrame, subset_indices: Union[Tuple[int, int], List[int], None]
) -> pd.DataFrame:
    """Extract a subset of the dataframe based on the provided indices. If no indices are provided, the entire dataframe is returned. Indices can be range or specific row indices."""
    if subset_indices is None:
        return df
    if isinstance(subset_indices, range):
        subset_indices = subset_indices
    if isinstance(subset_indices, tuple):
        subset_indices = range(*subset_indices)
    subset_df = df.iloc[subset_indices].copy()
    return subset_df


def get_compiled_regex_list(regex: Union[str, re.Pattern, List[re.Pattern]]) -> List[re.Pattern]:
    """Compile the regex pattern(s) provided and return a list of compiled regex patterns."""
    if isinstance(regex, str):
        return [re.compile(rf"{regex}")]
    elif isinstance(regex, re.Pattern):
        return [regex]
    elif isinstance(regex, list):
        return regex
    else:
        print("illegal version of regex")


def get_regex_match(response: str, regex_list: List[re.Pattern]) -> Union[str, None]:
    """Extract the first match from the response using the provided regex patterns. Return first match if multiple exist.
    Note: This function assumes the regex patterns each specify exactly 1 group that is the match group using '(<group>)'."""
    for regex_pattern in regex_list:
        pattern_match = regex_pattern.match(response)
        if pattern_match:
            return pattern_match.group(
                1
            )  # TODO: currently, this assumes 1 group in the supplied regex as the "match" group and takes that match if it exists.
    return None


def get_return_values_match(response: str, return_values_pattern: re.Pattern) -> Union[str, None]:
    """Extract the provided return values from the response using regex pattern. Return first extracted value if multiple exist."""
    exact_matches = re.findall(return_values_pattern, str(response))
    if len(exact_matches) > 0:
        return exact_matches[0]
    return None
