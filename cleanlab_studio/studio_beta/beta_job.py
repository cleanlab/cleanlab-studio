from __future__ import annotations

import enum
import itertools
import pathlib
import time
from dataclasses import dataclass
from typing import List, Optional

import requests
from tqdm import tqdm

from cleanlab_studio.errors import BetaJobError, DownloadResultsError
from cleanlab_studio.internal.api.beta_api import (
    get_job,
    get_job_status,
    get_results,
    list_jobs,
    run_job,
)


class JobStatus(enum.Enum):
    """The status of a job in the Cleanlab Studio Beta API."""

    CREATED = 0
    RUNNING = 1
    READY = 2
    FAILED = -1

    @classmethod
    def from_name(cls, name: str) -> "JobStatus":
        return cls[name.upper()]


@dataclass
class BetaJob:
    """Represents a job in the Cleanlab Studio Beta API."""

    id: str
    status: JobStatus
    dataset_id: str
    job_definition_name: str
    created_at: int
    _api_key: str

    @classmethod
    def from_id(cls, api_key: str, job_id: str) -> "BetaJob":
        """Loads an existing job by ID.
        Args:
            api_key: Your API key.
            job_id: The ID of the job to load.
        Returns:
            The job object.
        """
        job_resp = get_job(api_key, job_id)
        job = cls(
            _api_key=api_key,
            id=job_resp["id"],
            dataset_id=job_resp["dataset_id"],
            job_definition_name=job_resp["job_definition_name"],
            status=JobStatus.from_name(job_resp["status"]),
            created_at=job_resp["created_at"],
        )
        return job

    @classmethod
    def run(cls, api_key: str, dataset_id: str, job_definition_name: str) -> "BetaJob":
        """Creates and runs a new job with the given dataset and job definition. Raises an error if the job definition name is invalid.

        Args:
            api_key: Your API key.
            dataset_id: The ID of the dataset to run the job on.
            job_definition_name: The name of the job definition to run.
        """
        job_resp = run_job(api_key, dataset_id, job_definition_name)
        job = cls(
            _api_key=api_key,
            id=job_resp["id"],
            dataset_id=dataset_id,
            job_definition_name=job_definition_name,
            status=JobStatus.from_name(job_resp["status"]),
            created_at=job_resp["created_at"],
        )
        return job

    def wait_until_ready(self, timeout: Optional[int] = None) -> None:
        """Blocks until a job is ready or the timeout is reached.

        Args:
            timeout (Optional[float], optional): timeout for polling, in seconds. Defaults to None.

        Raises:
            TimeoutError: if job is not ready by end of timeout.
            BetaJobError: if job fails.
        """
        start_time = time.time()
        res = get_job_status(self._api_key, self.id)
        self.status = JobStatus.from_name(res["status"])
        spinner = itertools.cycle("|/-\\")

        with tqdm(
            total=JobStatus.READY.value,
            desc="Job Progress: \\",
            bar_format="{desc} {postfix}",
        ) as pbar:
            while self.status != JobStatus.READY and self.status != JobStatus.FAILED:
                pbar.set_postfix_str(self.status.name.capitalize())
                pbar.update(int(self.status.value) - pbar.n)

                if timeout is not None and time.time() - start_time > timeout:
                    raise TimeoutError("Result not ready before timeout")

                for _ in range(50):
                    time.sleep(0.1)
                    pbar.set_description_str(f"Job Progress: {next(spinner)}")

                res = get_job_status(self._api_key, self.id)
                self.status = JobStatus.from_name(res["status"])

            if self.status == JobStatus.READY:
                pbar.update(pbar.total - pbar.n)
                pbar.set_postfix_str(self.status.name.capitalize())
                return

            if self.status == JobStatus.FAILED:
                pbar.set_postfix_str(self.status.name.capitalize())
                raise BetaJobError(f"Experimental job {self.id} failed to complete")

    def download_results(self, output_filepath: str) -> None:
        """Downloads the results of an experimental job to the given output filepath.

        Args:
            output_filepath: The path to save the downloaded results to.
        Raises:
            BetaJobError: if job is not yet ready or has failed.
            DownloadResultsError: if output file extension does not match result file type.
        """
        output_path = pathlib.Path(output_filepath)

        if self.status == JobStatus.FAILED:
            raise BetaJobError("Job failed, cannot download results")

        if self.status != JobStatus.READY:
            raise BetaJobError("Job must be ready to download results")

        results = get_results(self._api_key, self.id)
        if output_path.suffix != results["result_file_type"]:
            raise DownloadResultsError(
                f"Output file extension does not match result file type {results['result_file_type']}"
            )

        resp = requests.get(results["result_url"])
        resp.raise_for_status()
        output_path.write_bytes(resp.content)

    @classmethod
    def list(cls, api_key: str) -> List[BetaJob]:
        """Lists all jobs you have run through the Beta API.

        Args:
            api_key: Your API key.
        Returns:
            A list of all the jobs you have run through the Cleanlab Studio Beta API.
        """
        jobs = list_jobs(api_key)
        return [
            cls(
                _api_key=api_key,
                id=job["id"],
                dataset_id=job["dataset_id"],
                job_definition_name=job["job_definition_name"],
                status=JobStatus.from_name(job["status"]),
                created_at=job["created_at"],
            )
            for job in jobs
        ]
