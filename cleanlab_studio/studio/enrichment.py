"""
Methods for interfacing with Enrichment Projects.

**This module is not meant to be imported and used directly.** Instead, use [`Studio.get_enrichment_project()`](../studio/#method-get_enrichment_project) to instantiate an [EnrichmentProject](#class-enrichmentproject) object.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, cast

import pandas as pd
from typing_extensions import NotRequired

from cleanlab_studio.internal.api import api
from cleanlab_studio.internal.types import JSONDict, TLMQualityPreset
from cleanlab_studio.studio.trustworthy_language_model import TLMOptions
from cleanlab_studio.internal.enrich_helpers import enrichment_ready, poll_enrichment_status


Replacement = Tuple[str, str]
RegexInput = Optional[Union[str, Replacement, List[Replacement]]]

ROW_ID_COLUMN_NAME = "row_id"
FINAL_RESULT_COLUMN_NAME = "final_result"
TRUSTWORTHINESS_SCORE_COLUMN_NAME = "trustworthiness_score"
LOG_COLUMN_NAME = "log"
RAW_RESULT_COLUMN_NAME = "raw_result"
REGEX_PARAMETER_ERROR_MESSAGE = (
    "The 'regex' parameter must be a string, a tuple(str, str), or a list of tuple(str, str)."
)
RUN_ALREADY_CALLED_ERROR_MESSAGE = "The run method can only be called once for each project, and it has already been called on this project."
RUN_NOT_CALLED_ERROR_MESSAGE = "The run method must be called before calling {funtion_name}."


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
        self._run_called = False  # Flag to check if run has been called on the entire dataset
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
    def _job_id(self) -> str:
        job_id: str = self._get_enrichment_project_dict()["id"]
        return job_id

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
    ) -> EnrichmentPreviewResult:
        """Enrich a subset of data for a preview."""
        extraction_pattern = None
        replacements: List[Dict[str, str]] = []

        _validate_enrichment_options(options)
        user_input_regex = options.get("regex")
        if user_input_regex:
            replacements = _get_regex_replacemenets_dictionary(user_input_regex)
            extraction_pattern = _get_extraction_pattern_dictionary(user_input_regex)

        response = api.enrichment_preview(
            api_key=self._api_key,
            project_id=self._id,
            new_column_name=new_column_name,
            constrain_outputs=options.get("constrain_outputs", None),
            extraction_pattern=extraction_pattern,
            indices=indices,
            optimize_prompt=options.get("optimize_prompt", True),
            prompt=options["prompt"],
            quality_preset=options.get("quality_preset", "medium"),
            replacements=replacements,
            tlm_options=cast(Dict[str, Any], options.get("tlm_options"))
            if options.get("tlm_options")
            else {},
        )
        enrichment_results = EnrichmentPreviewResult.from_dict(response)

        return enrichment_results

    def run(
        self,
        options: EnrichmentOptions,
        *,
        new_column_name: str,
    ) -> EnrichmentResult:
        """Enrich the full dataset."""
        if self._run_called:
            raise ValueError(RUN_ALREADY_CALLED_ERROR_MESSAGE)
        else:
            self._run_called = True

        extraction_pattern = None
        replacements: List[Dict[str, str]] = []

        _validate_enrichment_options(options)

        user_input_regex = options.get("regex")
        if user_input_regex:
            replacements = _get_regex_replacemenets_dictionary(user_input_regex)
            extraction_pattern = _get_extraction_pattern_dictionary(user_input_regex)

        response = api.run(
            api_key=self._api_key,
            project_id=self._id,
            new_column_name=new_column_name,
            constrain_outputs=options.get("constrain_outputs", None),
            extraction_pattern=extraction_pattern,
            optimize_prompt=options.get("optimize_prompt", True),
            prompt=options["prompt"],
            quality_preset=options.get("quality_preset", "medium"),
            replacements=replacements,
            tlm_options=cast(Dict[str, Any], options.get("tlm_options"))
            if options.get("tlm_options")
            else {},
        )
        enrichment_results = EnrichmentResult.from_dict(response)

        return enrichment_results

    def wait_until_ready(self) -> None:
        if not self._run_called:
            raise ValueError(RUN_NOT_CALLED_ERROR_MESSAGE.format(function_name="wait_until_ready"))
        else:
            return poll_enrichment_status(
                api_key=self._api_key,
                project_id=self._id,
                job_id=self._job_id,
            )

    def ready(self) -> bool:
        if not self._run_called:
            raise ValueError(RUN_NOT_CALLED_ERROR_MESSAGE.format(function_name="ready"))
        else:
            return enrichment_ready(
                api_key=self._api_key,
                job_id=self._job_id,
            )


class EnrichmentOptions(TypedDict):
    """Options for enriching a dataset with a Trustworthy Language Model (TLM).

    Args:
        prompt (str): Using string.Template, that contains both the prompt, and names of columns to embed.
            **Example:** "Is this a numeric value, answer Yes or No only. Value: ${column_name}"
        constrain_outputs (List[str], optional): List of all possible output values for the `metadata` column.
            If specified, every entry in the `metadata` column will exactly match one of these values (for less open-ended data enrichment tasks). If None, the `metadata` column can contain arbitrary values (for more open-ended data enrichment tasks).
           There may be additional transformations applied to ensure the returned value is one of these. If regex is also specified, then these transformations occur after your regex is applied.
            If `optimize_prompt` is True, the prompt will be automatically adjusted to include a statement that the response must match one of the `constrain_outputs`.
        optimize_prompt (bool, default = True): When False, your provided prompt will not be modified in any way. When True, your provided prompt may be automatically adjusted in an effort to produce better results.
            For instance, if the constrain_outputs are constrained, we may automatically append the following statement to your prompt: "Your answer must exactly match one of the following values: `constrain_outputs`."
        quality_preset (TLMQualityPreset, default = "medium"): The quality preset to use for the Trustworthy Language Model (TLM) to use for data enrichment.
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
        tlm_options (TLMOptions, default = {}): Options for the Trustworthy Language Model (TLM) to use for data enrichment.
    """

    prompt: str
    constrain_outputs: NotRequired[Optional[List[str]]]
    optimize_prompt: NotRequired[bool]
    quality_preset: NotRequired[TLMQualityPreset]
    regex: NotRequired[RegexInput]
    tlm_options: NotRequired[TLMOptions]


def _validate_enrichment_options(options: EnrichmentOptions) -> None:
    """Validate the enrichment options."""
    # Validate the prompt
    if len(options["prompt"]) == 0:
        raise ValueError("The 'prompt' parameter must be a non-empty string.")

    # Validate the regex
    def _validate_tuple_is_replacement(t: Tuple[Any, ...]) -> None:
        if isinstance(t, tuple) and len(t) == 2 and all(isinstance(x, str) for x in t):
            return None
        raise ValueError(REGEX_PARAMETER_ERROR_MESSAGE)

    if "regex" in options:
        user_input_regex = options["regex"]
        if user_input_regex:
            if isinstance(user_input_regex, str):
                return None
            elif isinstance(user_input_regex, tuple):
                _validate_tuple_is_replacement(user_input_regex)
            elif isinstance(user_input_regex, list):
                for replacement in user_input_regex:
                    _validate_tuple_is_replacement(replacement)
            else:
                raise ValueError(REGEX_PARAMETER_ERROR_MESSAGE)


def _get_regex_replacemenets_dictionary(user_input_regex: RegexInput) -> List[Dict[str, Any]]:
    replacements: List[Dict[str, str]] = []
    if isinstance(user_input_regex, tuple):
        replacements.append({"pattern": user_input_regex[0], "replacement": user_input_regex[1]})
    elif isinstance(user_input_regex, list):
        for replacement in user_input_regex:
            replacements.append({"pattern": replacement[0], "replacement": replacement[1]})
    return replacements


def _get_extraction_pattern_dictionary(user_input_regex: RegexInput) -> Optional[str]:
    extraction_pattern = None
    if isinstance(user_input_regex, str):
        extraction_pattern = user_input_regex

    return extraction_pattern


class EnrichmentResult:
    """Enrichment result."""

    def __init__(self, results: pd.DataFrame):
        self._results = results.sort_values(by=ROW_ID_COLUMN_NAME)

    @classmethod
    def from_dict(cls, json_dict: JSONDict) -> EnrichmentResult:
        raise NotImplementedError()

    def to_list(self) -> List[Tuple[Optional[str], float]]:
        raise NotImplementedError()

    def details(self) -> pd.DataFrame:
        return self._results

    def join(self, original_data: pd.DataFrame, *, with_details: bool = False) -> pd.DataFrame:
        df = self._results
        joined_data = original_data.join(df, how="left")
        return joined_data


class EnrichmentPreviewResult(EnrichmentResult):
    """Enrichment preview result."""

    _is_timeout: bool
    _failed_jobs_count: int
    _completed_jobs_count: int
    _final_result_column_name: str
    _trustworthiness_score_column_name: str

    @classmethod
    def from_dict(cls, json_dict: Dict[str, Any]) -> EnrichmentPreviewResult:
        new_column_name_mapping = json_dict["new_column_name_mapping"]

        # Prepare the results DataFrame from the 'results' list
        results = json_dict["results"]
        df = pd.DataFrame(results)

        # Set the index to row_id for easier joining later
        df.set_index(ROW_ID_COLUMN_NAME, inplace=True)

        # Select and rename the columns to match the expected format
        df = df[
            [
                FINAL_RESULT_COLUMN_NAME,
                TRUSTWORTHINESS_SCORE_COLUMN_NAME,
                RAW_RESULT_COLUMN_NAME,
                LOG_COLUMN_NAME,
            ]
        ]
        df.rename(
            columns={
                FINAL_RESULT_COLUMN_NAME: new_column_name_mapping[FINAL_RESULT_COLUMN_NAME],
                TRUSTWORTHINESS_SCORE_COLUMN_NAME: new_column_name_mapping[
                    TRUSTWORTHINESS_SCORE_COLUMN_NAME
                ],
                RAW_RESULT_COLUMN_NAME: new_column_name_mapping[RAW_RESULT_COLUMN_NAME],
                LOG_COLUMN_NAME: new_column_name_mapping[LOG_COLUMN_NAME],
            },
            inplace=True,
        )

        # Create an instance of EnrichmentPreviewResult
        instance = cls(results=df)

        # Set the additional attributes
        instance._is_timeout = json_dict["is_timeout"]
        instance._completed_jobs_count = json_dict["completed_jobs_count"]
        instance._failed_jobs_count = json_dict["failed_jobs_count"]
        instance._final_result_column_name = new_column_name_mapping[FINAL_RESULT_COLUMN_NAME]
        instance._trustworthiness_score_column_name = new_column_name_mapping[
            TRUSTWORTHINESS_SCORE_COLUMN_NAME
        ]

        return instance

    # undecided on whether to include this method
    def get_preview_status(self) -> Dict[str, bool | int]:
        """Get the status of the preview operation."""
        return {
            "is_timeout": self._is_timeout,
            "completed_jobs_count": self._completed_jobs_count,
            "failed_jobs_count": self._failed_jobs_count,
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
            df = self._results[
                [self._final_result_column_name, self._trustworthiness_score_column_name]
            ]
        joined_data = original_data.join(df, how="inner")

        return joined_data
