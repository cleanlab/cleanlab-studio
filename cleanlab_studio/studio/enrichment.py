"""
Methods for interfacing with Enrichment Projects.

**This module is not meant to be imported and used directly.** Instead, use [`Studio.get_enrichment_project()`](../studio/#method-get_enrichment_project) to instantiate an [EnrichmentProject](#class-enrichmentproject) object.
"""

from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, Optional, Union, List, TypedDict, Tuple

import pandas as pd

from cleanlab_studio.internal.api import api
from cleanlab_studio.internal.types import JSONDict
from cleanlab_studio.studio.trustworthy_language_model import TLMOptions

Replacement = Tuple[str, str]


def _response_timestamp_to_datetime(timestamp_string: str) -> datetime:
    """
    Converts the timestamp strings returned by the Cleanlab Studio API into datetime typing.
    """
    response_timestamp_format_str = "%a, %d %b %Y %H:%M:%S %Z"
    return datetime.strptime(timestamp_string, response_timestamp_format_str)


class EnrichmentProject:
    """Represents an Enrichment Project instance, which is bound to a Cleanlab Studio account.

    EnrichmentProjects should be instantiated using the [`Studio.get_enrichment_project()`](../studio/#method-get_enrichment_project) method.
    """

    def __init__(
        self,
        api_key: str,
        id: str,
        name: str,
        created_at: Optional[Union[str, datetime]] = None,
    ) -> None:
        """Initialize an EnrichmentProject.

        **Objects of this class are not meant to be constructed directly.**
        Instead, use [`Studio.get_enrichment_project()`](../studio/#method-get_enrichment_project).
        """
        self._api_key = api_key
        self._id = id
        self._name = name
        self._created_at: Optional[datetime]
        if isinstance(created_at, str):
            self._created_at = _response_timestamp_to_datetime(created_at)
        else:
            self._created_at = created_at

    def _get_enrichment_project_dict(self) -> Dict[str, Any]:
        return dict(api.get_enrichment_project(api_key=self._api_key, project_id=self._id))

    @property
    def id(self) -> str:
        """
        (str) ID of the Enrichment Project.
        """
        return self._id

    @property
    def name(self) -> str:
        """
        (str) Name of the Enrichment Project.
        """
        return self._name

    @property
    def created_at(self) -> datetime:
        """
        (datetime.datetime) When the Enrichment Project was created.
        """
        if self._created_at is None:
            create_at_string = self._get_enrichment_project_dict()["created_at"]
            self._created_at = _response_timestamp_to_datetime(create_at_string)

        return self._created_at

    @property
    def updated_at(self) -> datetime:
        """
        (datetime.datetime) When the Enrichment Project was last updated.
        """
        updated_at = self._get_enrichment_project_dict()["updated_at"]
        return _response_timestamp_to_datetime(updated_at)

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary of EnrichmentProject metadata.
        """
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def preview(
        self,
        options: EnrichmentOptions,
        *,
        new_column_name: str,
        indices: Optional[List[int]] = None,
        disable_warnings: bool = False,
    ) -> EnrichmentPreviewResult:
        response = api.enrichment_preview(
            api_key=self._api_key,
            project_id=self._id,
            options=options,
            new_column_name=new_column_name,
            indices=indices,
        )
        epr = EnrichmentPreviewResult.from_dict(response)
        if not disable_warnings and epr._is_timeout:
            print(
                "Warning: The preview operation timed out for a subset of data. Set those results to None."
            )
        return epr

class EnrichmentOptions(TypedDict):
    """Options for enriching a dataset with a Trustworthy Language Model (TLM).

    ref: https://github.com/cleanlab/cleanlab-studio/blob/main/cleanlab_studio/utils/data_enrichment/enrich.py#L34

    Args:
        prompt (str): Formatted f-string, that contains both the prompt, and names of columns to embed.
            **Example:** "Is this a numeric value, answer Yes or No only. Value: {column_name}"
        constrain_outputs (List[str], optional): List of all possible output values for the `metadata` column.
            If specified, every entry in the `metadata` column will exactly match one of these values (for less open-ended data enrichment tasks). If None, the `metadata` column can contain arbitrary values (for more open-ended data enrichment tasks).
           There may be additional transformations applied to ensure the returned value is one of these. If regex is also specified, then these transformations occur after your regex is applied.
            If `optimize_prompt` is True, the prompt will be automatically adjusted to include a statement that the response must match one of the `constrain_outputs`.
        optimize_prompt (bool, default = True): When False, your provided prompt will not be modified in any way. When True, your provided prompt may be automatically adjusted in an effort to produce better results.
            For instance, if the constrain_outputs are constrained, we may automatically append the following statement to your prompt: "Your answer must exactly match one of the following values: `constrain_outputs`."
        replacements (str | Replacement | List[Replacement], optional): A string, tuple, or list of tuples specifying regular expressions to apply for post-processing the raw LLM outputs.

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
        tlm_options (TLMOptions, optional): Options for the Trustworthy Language Model (TLM) to use for data enrichment.
    """

    prompt: str
    constrain_outputs: Optional[List[str]]
    optimize_prompt: Optional[bool]
    replacements: Optional[Union[str, Replacement, List[Replacement]]]
    tlm_options: Optional[TLMOptions]


class EnrichmentResult:
    def __init__(self, results: pd.DataFrame, new_column_name: str):
        self._results = results
        self._new_column_name = new_column_name

    @classmethod
    def from_dict(cls, json_dict: JSONDict) -> EnrichmentResult:
        raise NotImplementedError()

    def to_list(self) -> List[Tuple[Optional[str], float]]:
        raise NotImplementedError()

    def details(self) -> pd.DataFrame:
        return self._results

    def join(self, original_data: pd.DataFrame, *, with_details: bool = False) -> pd.DataFrame:
        df = self._results
        if not with_details:
            df = self._results[[self._new_column_name]]

        joined_data = original_data.join(df, how="left")
        return joined_data


class EnrichmentPreviewResult(EnrichmentResult):
    _indices: List[int]
    _errors: Dict
    _is_timeout: bool
    _total_count: int
    _successful_count: int
    _failed_count: int

    @classmethod
    def from_dict(cls, json_dict: Dict) -> EnrichmentPreviewResult:
        # Extract the new column name from the response
        new_column_name = json_dict["new_column_name"]

        # Prepare the results DataFrame from the 'results' list
        results = json_dict["results"]
        df = pd.DataFrame(results)

        # Set the index to row_id for easier joining later
        df.set_index("row_id", inplace=True)

        # Select and rename the columns to match the expected format
        df = df[["final_result", "trustworthy_score", "log"]]
        df.rename(columns={
            "final_result": new_column_name,
            "trustworthy_score": f"{new_column_name}_trustworthy_score",
            "log": f"{new_column_name}_log"
        }, inplace=True)

        # Create an instance of EnrichmentPreviewResult
        instance = cls(
            results=df,
            new_column_name=new_column_name
        )

        # Set the additional attributes
        instance._indices = df.index.tolist()
        instance._errors = json_dict.get("errors", {})
        instance._is_timeout = json_dict["is_timeout"]
        instance._total_count = len(results)
        instance._successful_count = json_dict["completed_jobs_count"]
        instance._failed_count = json_dict["failed_jobs_count"]

        return instance
    
    def get_preview_status(self) -> Dict:
        return {
            "is_timeout": self._is_timeout,
            "total_count": self._total_count,
            "successful_count": self._successful_count,
            "failed_count": self._failed_count,
        }
    
    def join(self, original_data: pd.DataFrame, *, with_details: bool = False) -> pd.DataFrame:
        """Join the original data with the enrichment results. 
        The result only contains those rows that were enriched by preview.
        
        Args:
            original_data (pd.DataFrame): The original data to join with the enrichment results.
            with_details (bool): If `with_details` is True, the details of the enrichment results will be included in the output DataFrame.
        """
        df = self._results
        if not with_details:
            df = self._results[[self._new_column_name]]
        joined_data = original_data.join(df, how="inner")

        return joined_data
