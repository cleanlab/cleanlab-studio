from typing import List

from cleanlab_studio.internal.studio_base import StudioBase
from cleanlab_studio.studio_beta.beta_dataset import BetaDataset
from cleanlab_studio.studio_beta.beta_job import BetaJob


class StudioBeta(StudioBase):
    def __init__(self, api_key: str):
        """
        Creates a client to interact with the Cleanlab Studio Beta API.
        Args:
            api_key: You can find your API key on your [account page](https://app.cleanlab.ai/account) in Cleanlab Studio. Instead of specifying the API key here, you can also log in with `cleanlab login` on the command-line.
        """
        super().__init__(api_key)

    def upload_dataset(
        self,
        filepath: str,
    ) -> BetaDataset:
        """Uploads a dataset from the given filepath for use in the Cleanlab Studio Beta API.
        Args:
            filepath: The path to the dataset file.

        Returns:
            The uploaded dataset object. You can use this object to obtain the ID of the uploaded dataset.
            Use this ID to run jobs on the dataset.
        """
        return BetaDataset.from_filepath(self._api_key, filepath)

    def run_job(self, dataset_id: str, job_definition_name: str) -> BetaJob:
        """Runs a Cleanlab Studio Beta job with the given dataset and job definition.
        Args:
            dataset_id: The ID of the dataset to run the job on.
            job_definition_name: The name of the job definition to run.

        Returns:
            The object representing the job. You can use this object to check the status of the job and download the results.
        """
        return BetaJob.run(self._api_key, dataset_id, job_definition_name)

    def download_results(self, job_id: str, output_filename: str) -> None:
        """Downloads the results of an experimental job to the given output filename.
        Args:
            job_id: The ID of the job to download the results for.
            output_filename: The path to save the downloaded results to.
        """
        BetaJob.from_id(self._api_key, job_id).download_results(output_filename)

    def list_datasets(self) -> List[BetaDataset]:
        """Lists all datasets you have uploaded through the Beta API.

        Returns:
            A list of all the datasets you have uploaded through the Cleanlab Studio Beta API.
        """
        return BetaDataset.list(self._api_key)

    def list_jobs(self) -> List[BetaJob]:
        """Lists all jobs you have run through the Beta API.

        Returns:
            A list of all the jobs you have run through the Cleanlab Studio Beta API.
        """
        return BetaJob.list(self._api_key)
