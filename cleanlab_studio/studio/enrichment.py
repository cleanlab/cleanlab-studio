"""
Methods for interfacing with Enrichment Projects.

**This module is not meant to be imported and used directly.** Instead, use [`Studio.get_enrichment_project()`](../studio/#method-get_enrichment_project) to instantiate an [EnrichmentProject](#class-enrichmentproject) object.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, cast

import matplotlib.pyplot as plt
import pandas as pd
from typing_extensions import NotRequired

from cleanlab_studio.internal.api import api
from cleanlab_studio.internal.types import JSONDict, TLMQualityPreset
from cleanlab_studio.studio.trustworthy_language_model import TLMOptions

Replacement = Tuple[str, str]
ROW_ID_COLUMN_NAME = "row_id"
FINAL_RESULT_COLUMN_NAME = "final_result"
TRUSTWORTHINESS_SCORE_COLUMN_NAME = "trustworthiness_score"
LOG_COLUMN_NAME = "log"
RAW_RESULT_COLUMN_NAME = "raw_result"
REGEX_PARAMETER_ERROR_MESSAGE = (
    "The 'regex' parameter must be a string, a tuple(str, str), or a list of tuple(str, str)."
)
CLEANLAB_ROW_ID_COLUMN_NAME = "cleanlab_row_ID"
CHECK_READY_INTERVAL = 60 * 2  # 2 minutes


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
        self._latest_populate_job: EnrichmentJob | None = None
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
    ) -> EnrichmentPreviewResults:
        """Enrich a subset of data for a preview."""
        _validate_enrichment_options(options)

        user_input_regex = options.get("regex")
        extraction_pattern, replacements = _handle_replacements_and_extraction_pattern(
            user_input_regex
        )

        self._latest_populate_job = None
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
            tlm_options=(
                cast(Dict[str, Any], options.get("tlm_options"))
                if options.get("tlm_options")
                else {}
            ),
        )
        epr = EnrichmentPreviewResults.from_dict(response["results"])

        return epr

    def run(
        self,
        options: EnrichmentOptions,
        *,
        new_column_name: str,
    ) -> dict[str, Any]:
        """Enrich the entire dataset."""
        _validate_enrichment_options(options)

        user_input_regex = options.get("regex")
        extraction_pattern, replacements = _handle_replacements_and_extraction_pattern(
            user_input_regex
        )

        response = api.enrichment_populate(
            api_key=self._api_key,
            project_id=self._id,
            new_column_name=new_column_name,
            constrain_outputs=options.get("constrain_outputs", None),
            extraction_pattern=extraction_pattern,
            optimize_prompt=options.get("optimize_prompt", True),
            prompt=options["prompt"],
            quality_preset=options.get("quality_preset", "medium"),
            replacements=replacements,
            tlm_options=(
                cast(Dict[str, Any], options.get("tlm_options"))
                if options.get("tlm_options")
                else {}
            ),
        )
        return response

    @property
    def ready(self) -> bool:
        """Check if the latest populate job is ready."""
        latest_job = self.list_all_jobs()[0]
        if latest_job["job_type"] != "ENRICHMENT":
            raise ValueError(
                "The latest job is a preview, to execute against entire dataset, please do `run()` first."
            )
        self.latest_populate_job = latest_job
        if latest_job["status"] == "FAILED":
            raise ValueError("The latest populate job failed.")
        elif latest_job["status"] == "RUNNING":
            return False
        elif latest_job["status"] == "CREATED":
            return False
        elif latest_job["status"] == "SUCCEEDED":
            return True
        else:
            raise ValueError("The latest populate job has an unknown status.")

    def wait_until_ready(self) -> None:
        """Wait until the latest populate job is ready."""
        while not self.ready:
            time.sleep(CHECK_READY_INTERVAL)
            pass

    def download_results(
        self, job_id: Optional[str] = None, include_original_dataset: Optional[bool] = False
    ) -> EnrichmentResults:
        """Get the results of a populate job."""
        self._latest_populate_job = self.list_all_jobs()[0]
        latest_job_id = job_id or self._latest_populate_job["id"]

        page = 1
        results = []
        resp = api.get_enrichment_job_result(
            api_key=self._api_key,
            job_id=latest_job_id,
            page=page,
            include_original_dataset=include_original_dataset,
        )
        results.extend(resp)

        while resp:
            page += 1
            resp = api.get_enrichment_job_result(
                api_key=self._api_key,
                job_id=latest_job_id,
                page=page,
                include_original_dataset=include_original_dataset,
            )
            results.extend(resp)

        return EnrichmentResults.from_dict(
            results, include_original_dataset=include_original_dataset
        )

    def list_all_jobs(self) -> List[EnrichmentJob]:
        """List all jobs in the project."""
        jobs = api.list_enrichment_jobs(api_key=self._api_key, project_id=self._id)
        typed_jobs = []
        for job in jobs:
            enrichment_options_dict = dict(
                prompt=job["prompt"],
                constrain_outputs=job.get("constrain_outputs"),
                optimize_prompt=job.get("optimize_prompt"),
                quality_preset=job.get("quality_preset"),
                regex=job.get("regex"),
                tlm_options=job.get("tlm_options"),
            )

            enrichment_options_dict = {
                k: v for k, v in enrichment_options_dict.items() if v is not None
            }

            enrichment_job = EnrichmentJob(
                id=job["id"],
                status=job["status"],
                created_at=_response_timestamp_to_datetime(job["created_at"]),
                updated_at=_response_timestamp_to_datetime(job["updated_at"]),
                enrichment_options=EnrichmentOptions(**enrichment_options_dict),  # type: ignore
                average_trustworthiness_score=job["average_trustworthiness_score"],
                job_type=job["type"],
                new_column_name=job["new_column_name"],
                indices=job.get("indices"),
            )
            typed_jobs.append(enrichment_job)

        self._latest_populate_job = typed_jobs[0]
        return typed_jobs

    def show_trustworthiness_score_history(self) -> None:
        """Show the trustworthiness score history of all jobs in the project."""
        data = self.list_all_jobs()
        self._latest_populate_job = data[0]
        data_sorted = sorted(data, key=lambda x: x["created_at"])
        scores = []
        dates = []

        for entry in data_sorted:
            score = entry["average_trustworthiness_score"]
            created_at = entry["created_at"].strftime("%Y-%m-%d %H:%M:%S")

            if score is not None:
                scores.append(score)
                dates.append(created_at)

        plt.figure(figsize=(10, 5))
        plt.plot(range(len(scores)), scores, marker="o", linestyle="-", color="b")
        plt.xlabel("Time (Ordered Events)")
        plt.ylabel("Average Trustworthiness Score")
        plt.title("Average Trustworthiness Score Over Time (Evenly Spaced)")
        plt.grid(True)
        plt.xticks(range(len(dates)), dates, rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def export_results_as_csv(self, job_id: Optional[str] = None) -> None:
        """Download the results of a job."""
        self._latest_populate_job = self.list_all_jobs()[0]
        latest_job_id = job_id or self._latest_populate_job["id"]

        file_name = api.export_results(api_key=self._api_key, job_id=latest_job_id)
        print(f"Results exported successfully at ./{file_name}")


class EnrichmentJob(TypedDict):
    """Represents an Enrichment Job instance.

    **This class is not meant to be constructed directly.** Instead, use the `EnrichmentProject` methods to create and manage Enrichment Jobs.
    """

    id: str
    status: str
    created_at: datetime
    updated_at: datetime
    enrichment_options: EnrichmentOptions
    average_trustworthiness_score: float
    job_type: str
    new_column_name: str
    indices: Optional[List[int]]


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
    constrain_outputs: NotRequired[List[str]]
    optimize_prompt: NotRequired[bool]
    quality_preset: NotRequired[TLMQualityPreset]
    regex: NotRequired[Union[str, Replacement, List[Replacement]]]
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


class EnrichmentResults:
    """Enrichment result."""

    _detailed_column_names: List[str]
    _include_original_dataset: bool

    def __init__(self, results: pd.DataFrame):
        self._results = results

    @classmethod
    def from_dict(
        cls, json_dict: List[JSONDict], include_original_dataset: Optional[bool] = False
    ) -> EnrichmentResults:
        df = pd.DataFrame(json_dict)
        df.set_index(CLEANLAB_ROW_ID_COLUMN_NAME, inplace=True)

        # cleanlab_row_ID is the row ID of the original data + 1. so need to change to 0-based index
        df.index = df.index - 1
        df.index.name = None
        instance = cls(results=df)

        new_column_names = _find_pattern_columns(df)

        instance._detailed_column_names = [f"{col}_raw" for col in new_column_names] + [
            f"{col}_log" for col in new_column_names
        ]
        instance._include_original_dataset = include_original_dataset or False
        return instance

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> EnrichmentResults:
        return cls(results=df)

    def details(self) -> pd.DataFrame:
        return self._results

    def join(self, original_data: pd.DataFrame, *, with_details: bool = False) -> pd.DataFrame:
        if self._include_original_dataset:
            raise ValueError(
                "The current results already contain the original data. You can get the joined data by calling `details()` method."
            )

        df = self._results
        joined_data = original_data.join(df, how="left")
        if not with_details:
            joined_data = joined_data.drop(columns=self._detailed_column_names)
        return joined_data


def _find_pattern_columns(df) -> List[str]:
    """Find the columns that match the pattern of the enrichment"""
    pattern = re.compile(r"(.+)(_trustworthiness_score|_raw|_log)?$")
    column_groups = {}  # type: Dict[str, List[str]]

    for col in df.columns:
        match = pattern.match(col)
        if match:
            base_col = match.group(1)
            if base_col not in column_groups:
                column_groups[base_col] = []
            column_groups[base_col].append(col)

    # Filter out groups that don't have all 4 expected columns
    valid_groups = {k: v for k, v in column_groups.items() if len(v) == 4}

    return list(valid_groups.keys())


class EnrichmentPreviewResults(EnrichmentResults):
    """Enrichment preview results."""

    @classmethod
    def from_dict(
        cls, json_dict: List[JSONDict], include_original_dataset: Optional[bool] = False
    ) -> EnrichmentPreviewResults:
        df = pd.DataFrame(json_dict)
        df.set_index(ROW_ID_COLUMN_NAME, inplace=True)
        df.sort_index(inplace=True)
        # Create an instance of EnrichmentPreviewResult
        instance = cls(results=df)
        new_column_names = _find_pattern_columns(df)

        instance._detailed_column_names = [f"{col}_raw" for col in new_column_names] + [
            f"{col}_log" for col in new_column_names
        ]
        instance._include_original_dataset = include_original_dataset or False
        return instance

    def join(self, original_data: pd.DataFrame, *, with_details: bool = False) -> pd.DataFrame:
        """Join the original data with the enrichment results.
        The result only contains those rows that were enriched by preview.

        Args:
            original_data (pd.DataFrame): The original data to join with the enrichment results.
            with_details (bool): If `with_details` is True, the details of the enrichment results will be included in the output DataFrame.
        """
        df = self._results
        joined_data = original_data.join(df, how="inner")
        if not with_details:
            joined_data = joined_data.drop(columns=self._detailed_column_names)

        return joined_data


def _handle_replacements_and_extraction_pattern(
    user_input_regex: Union[str, Replacement, List[Replacement], None]
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
            raise ValueError(REGEX_PARAMETER_ERROR_MESSAGE)
    return extraction_pattern, replacements
