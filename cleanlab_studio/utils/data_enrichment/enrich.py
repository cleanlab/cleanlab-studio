from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from cleanlab_studio.studio.enrichment import (
    EnrichmentProject, 
    EnrichmentOptions, 
    EnrichmentResults
)
import re

from cleanlab_studio.internal.enrichment_utils import (
    get_prompt_outputs,
    get_regex_match_or_replacement,
    get_constrain_outputs_match,
    get_optimized_prompt,
)

class ClientEnrichmentProject(EnrichmentProject):
    """Client-side extension of EnrichmentProject with online inference capability"""

    def __init__(self, api_key: str, id: str, name: str):
        super().__init__(api_key, id, name)

    def online_inference(
        self,
        data: pd.DataFrame,
        options: EnrichmentOptions,
        new_column_name: str,
    ) -> EnrichmentResults:
        """Process new data using the same enrichment logic as run().

        Args:
            data (pd.DataFrame): The new data to enrich.
            options (EnrichmentOptions): Options for enriching the dataset.
            new_column_name (str): The name of the new column to store the prompt results.

        Returns:
            EnrichmentResults: The results of the enrichment process.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for online inference. Install it with: pip install pandas"
            )

        # Validate inputs
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if "prompt" not in options:
            raise ValueError("options must contain 'prompt' key")

        # Process options
        prompt = (
            get_optimized_prompt(options["prompt"], options.get("constrain_outputs"))
            if options.get("optimize_prompt", True)
            else options["prompt"]
        )

        # Get TLM options
        tlm_kwargs = options.get("tlm_options", {}).copy()
        if "quality_preset" in options:
            tlm_kwargs["quality_preset"] = options["quality_preset"]

        # Process data
        outputs = get_prompt_outputs(api_key=self._api_key, prompt=prompt, data=data, studio=self, **tlm_kwargs)

        # Format results to match run() output
        results = pd.DataFrame()
        results[f"{new_column_name}_trustworthiness_score"] = [
            out["trustworthiness_score"] if out else None for out in outputs
        ]
        results[new_column_name] = [out["response"] if out else None for out in outputs]

        # Apply post-processing
        user_input_regex = options.get("regex")
        extraction_pattern, replacements = self._handle_replacements_and_extraction_pattern(
            user_input_regex
        )
        constrain_outputs = options.get("constrain_outputs")

        if extraction_pattern or replacements or constrain_outputs:
            results[f"{new_column_name}_raw"] = results[new_column_name].copy()
            results[f"{new_column_name}_log"] = results[new_column_name].copy()

            if extraction_pattern:
                results[new_column_name] = results[new_column_name].apply(
                    lambda x: get_regex_match_or_replacement(x, extraction_pattern)
                )

            for replacement in replacements:
                results[new_column_name] = results[new_column_name].apply(
                    lambda x: re.sub(replacement["pattern"], replacement["replacement"], str(x))
                )

            if constrain_outputs:
                results[new_column_name] = results[new_column_name].apply(
                    lambda x: get_constrain_outputs_match(x, constrain_outputs)
                )

        return EnrichmentResults.from_dataframe(results)

    @staticmethod
    def _handle_replacements_and_extraction_pattern(
        user_input_regex: Union[str, Tuple[str, str], List[Tuple[str, str]], None]
    ) -> Tuple[Optional[str], List[Dict[str, str]]]:
        extraction_pattern = None
        replacements: List[Dict[str, str]] = []

        if user_input_regex:
            if isinstance(user_input_regex, str):
                extraction_pattern = user_input_regex
            elif isinstance(user_input_regex, tuple):
                replacements.append(
                    {"pattern": user_input_regex[0], "replacement": user_input_regex[1]}
                )
            elif isinstance(user_input_regex, list):
                for replacement in user_input_regex:
                    replacements.append({"pattern": replacement[0], "replacement": replacement[1]})
            else:
                raise ValueError("Invalid regex format")
        return extraction_pattern, replacements

    def run(
        self,
        options: EnrichmentOptions,
        new_column_name: str,
    ) -> EnrichmentResults:
        """Enrich the entire dataset using the provided prompt."""
        # This method should be implemented to match the functionality in EnrichmentProject
        # For now, we'll raise a NotImplementedError
        raise NotImplementedError("run() method is not implemented in ClientEnrichmentProject")

# The process_regex function remains unchanged
def process_regex(
    column_data: Union["pd.Series", List[str]],
    regex: Union[str, Tuple[str, str], List[Tuple[str, str]]],
) -> Union["pd.Series", List[str]]:
    """
    Performs regex matches or replacements to the given string according to the given matching patterns.

    Args:
        column_data: A pandas Series or list of strings to process
        regex: String pattern for matching or tuple(s) for replacement

    Returns:
        Processed strings after applying regex operations
    """
    if isinstance(column_data, list):
        return [
            ClientEnrichmentProject._handle_replacements_and_extraction_pattern(x, regex)
            for x in column_data
        ]

    try:
        import pandas as pd

        if isinstance(column_data, pd.Series):
            return column_data.apply(
                lambda x: ClientEnrichmentProject._handle_replacements_and_extraction_pattern(
                    x, regex
                )
            )
    except ImportError:
        pass

    raise TypeError("column_data should be a pandas Series or a list of strings.")
