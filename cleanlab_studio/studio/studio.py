"""
Python API for Cleanlab Studio.
"""

from typing import Any, List, Literal, Optional, Union
from types import FunctionType
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import inference
from . import trustworthy_language_model
from cleanlab_studio.utils import tlm_hybrid
from cleanlab_studio.errors import CleansetError
from cleanlab_studio.internal import clean_helpers, deploy_helpers, upload_helpers
from cleanlab_studio.internal.api import api
from cleanlab_studio.internal.util import (
    init_dataset_source,
    telemetry,
    apply_corrections_snowpark_df,
    apply_corrections_spark_df,
    apply_corrections_pd_df,
)
from cleanlab_studio.internal.settings import CleanlabSettings
from cleanlab_studio.internal.types import SchemaOverride, TLMQualityPreset
from cleanlab_studio.errors import VersionError, MissingAPIKeyError, InvalidDatasetError

_snowflake_exists = api.snowflake_exists
if _snowflake_exists:
    import snowflake.snowpark as snowpark

_pyspark_exists = api.pyspark_exists
if _pyspark_exists:
    import pyspark.sql


class Studio:
    _api_key: str

    def __init__(self, api_key: Optional[str]):
        """
        Creates a Cleanlab Studio client.

        Args:
            api_key: You can find your API key on your [account page](https://app.cleanlab.ai/account) in Cleanlab Studio. Instead of specifying the API key here, you can also log in with `cleanlab login` on the command-line.

        """
        if not api.is_valid_client_version():
            raise VersionError(
                "CLI is out of date and must be updated. Run 'pip install --upgrade cleanlab-studio'."
            )
        if api_key is None:
            try:
                api_key = CleanlabSettings.load().api_key
                if api_key is None:
                    raise ValueError
            except (FileNotFoundError, KeyError, ValueError):
                raise MissingAPIKeyError(
                    "No API key found; either specify API key or log in with 'cleanlab login' first"
                )
        if not api.validate_api_key(api_key):
            raise ValueError(
                f"Invalid API key, please check if it is properly specified: {api_key}"
            )

        self._api_key = api_key

    def upload_dataset(
        self,
        dataset: Any,
        dataset_name: Optional[str] = None,
        *,
        schema_overrides: Optional[List[SchemaOverride]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Uploads a dataset to Cleanlab Studio.

        Args:
            dataset: Object representing the dataset to upload. Currently supported formats include a `str` path to your dataset, a pandas, snowflake, or pyspark DataFrame.
            dataset_name: Name for your dataset in Cleanlab Studio (optional if uploading from filepath).
            schema_overrides: Optional list of overrides you would like to make to the schema of your dataset. If not provided, all columns will be untyped. Format defined [here](/guide/concepts/datasets/#schema-updates).
            modality: [DEPRECATED] Optional parameter to override the modality of your dataset. If not provided, modality will be inferred.
            id_column: [DEPRECATED] Optional parameter to override the ID column of your dataset. If not provided, a monotonically increasing ID column will be generated.

        Returns:
            ID of uploaded dataset.
        """
        ds = init_dataset_source(dataset, dataset_name)
        if kwargs.get("modality") is not None:
            warnings.warn(
                "Ignoring `modality` parameter which is deprecated and will be removed in a future release.",
                FutureWarning,
            )
        if kwargs.get("id_column") is not None:
            warnings.warn(
                "Ignoring `id_column` parameter which is deprecated and will be removed in a future release.",
                FutureWarning,
            )

        if isinstance(schema_overrides, dict):
            # TODO: link to documentation for schema override format
            schema_overrides = upload_helpers.convert_schema_overrides(schema_overrides)
            warnings.warn(
                "Using deprecated `schema_overrides` format. Please use list of SchemaOverride objects instead.",
                FutureWarning,
            )

        return upload_helpers.upload_dataset(
            self._api_key,
            ds,
            schema_overrides=schema_overrides,
        )

    def delete_dataset(
        self,
        dataset_id: str,
    ) -> None:
        """Deletes a dataset from Cleanlab Studio.

        If the dataset is used in projects, the projects will be deleted as well.
        """
        api.delete_dataset(
            self._api_key,
            dataset_id,
        )

    def download_cleanlab_columns(
        self,
        cleanset_id: str,
        include_cleanlab_columns: bool = True,
        include_project_details: bool = False,
        to_spark: bool = False,
    ) -> Any:
        """
        Downloads [Cleanlab columns](/guide/concepts/cleanlab_columns/) for a cleanset.

        Args:
            cleanset_id: ID of cleanset to download columns from. To obtain cleanset ID from project ID use, [get_latest_cleanset_id](#method-get_latest_cleanset_id).
            include_cleanlab_columns: whether to download all Cleanlab columns or just the clean_label column
            include_project_details: whether to download columns related to project status such as resolved rows, actions taken, etc.

        Returns:
            A pandas or pyspark DataFrame. Type is `Any` to avoid requiring pyspark installation.
        """
        rows_df = api.download_cleanlab_columns(
            self._api_key,
            cleanset_id,
            include_cleanlab_columns=include_cleanlab_columns,
            include_project_details=include_project_details,
            to_spark=to_spark,
        )
        return rows_df

    def apply_corrections(self, cleanset_id: str, dataset: Any, keep_excluded: bool = False) -> Any:
        """
        Applies corrections from a Cleanlab Studio cleanset to your dataset. This function takes in your local copy of the original dataset, as well as the `cleanset_id` for the cleanset generated from this dataset in the Project web interface. The function returns a copy of your original dataset, where the label column has been substituted with corrected labels that you selected (either manually or via auto-fix) in the Cleanlab Studio web interface Project, and the rows you marked as excluded will be excluded from the returned copy of your original dataset. Corrections should have been made by viewing your Project in the Cleanlab Studio web interface (see [Cleanlab Studio web quickstart](/guide/quickstart/web#review-issues-detected-in-your-dataset-and-correct-them)).

        The intended workflow is: create a Project, correct your Dataset automatically/manually in the web interface to generate a Cleanset (cleaned dataset), then call this function to make your original dataset locally look like the current Cleanset.

        Args:
            cleanset_id: ID of cleanset to apply corrections from.
            dataset: Dataset to apply corrections to. Supported formats include pandas, snowpark, and pyspark DataFrame. Dataset should have the same number of rows as the dataset used to create the project. It should also contain a label column with the same name as the label column for the project.
            keep_excluded: Whether to retain rows with an "exclude" action. By default these rows will be removed from the dataset.

        Returns:
            A copy of the dataset with corrections applied.
        """
        project_id = api.get_project_of_cleanset(self._api_key, cleanset_id)
        label_col = api.get_label_column_of_project(self._api_key, project_id)
        id_col = api.get_id_column(self._api_key, cleanset_id)

        if _snowflake_exists and isinstance(dataset, snowpark.DataFrame):
            cl_cols = self.download_cleanlab_columns(
                cleanset_id, to_spark=False, include_project_details=True
            )
            return apply_corrections_snowpark_df(dataset, cl_cols, id_col, label_col, keep_excluded)

        elif _pyspark_exists and isinstance(dataset, pyspark.sql.DataFrame):
            cl_cols = self.download_cleanlab_columns(
                cleanset_id, to_spark=True, include_project_details=True
            )
            return apply_corrections_spark_df(dataset, cl_cols, id_col, label_col, keep_excluded)

        elif isinstance(dataset, pd.DataFrame):
            cl_cols = self.download_cleanlab_columns(cleanset_id, include_project_details=True)
            return apply_corrections_pd_df(dataset, cl_cols, id_col, label_col, keep_excluded)

        else:
            raise InvalidDatasetError(
                f"Provided unsupported dataset of type: {type(dataset)}. We currently support applying corrections to pandas or pyspark dataframes"
            )

    def create_project(
        self,
        dataset_id: str,
        project_name: str,
        modality: Literal["text", "tabular", "image"],
        *,
        task_type: Optional[
            Literal["multi-class", "multi-label", "regression", "unsupervised"]
        ] = "multi-class",
        model_type: Literal["fast", "regular"] = "regular",
        label_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        text_column: Optional[str] = None,
    ) -> str:
        """
        Creates a Cleanlab Studio project.

        Args:
            dataset_id: ID of dataset to create project for.
            project_name: Name for resulting project.
            modality: Modality of project (i.e. text, tabular, image).
            task_type: Type of ML task to perform. Select a supervised task type (i.e. "multi-class", "multi-label", "regression") if your dataset
                has a label column you would like to predict values for or detect erroneous values in. Select "unsupervised" if your dataset has
                no specific label column. See the [Projects Guide](/guide/concepts/projects/#machine-learning-task--dataset-type) for more
                information on task types.
            model_type: Type of model to train (i.e. "fast", "regular"). See the [Projects Guide](/guide/concepts/projects/#model-type) for more information on model types.
            label_column: Name of column in dataset containing labels (if not supplied, we'll make our best guess). For "unsupervised" tasks, this should be `None`.
            feature_columns: List of columns to use as features for a tabular project. By default all columns are used as feature columns. This parameter is particularly useful if your dataset has a column containing unique IDs and you want to exclude that column from the feature columns.
            text_column: Name of column containing the text to train text modality project on (if not supplied and modality is "text" we'll make our best guess).

        Returns:
            ID of created project.
        """
        dataset_details = api.get_dataset_details(self._api_key, dataset_id, task_type)

        if label_column is None and task_type is not None and task_type != "unsupervised":
            label_column = str(dataset_details["label_column_guess"])
            print(f"Label column not supplied. Using best guess {label_column}")

        if feature_columns is not None and modality != "tabular":
            if label_column in feature_columns:
                raise InvalidDatasetError("Label column cannot be included in feature columns")
            raise InvalidDatasetError(
                "Feature columns supplied, but project modality is not tabular"
            )
        if feature_columns is None:
            if modality == "tabular":
                feature_columns = dataset_details["distinct_columns"]
                if label_column is not None and label_column in feature_columns:
                    feature_columns.remove(label_column)
                print(f"Feature columns not supplied. Using all valid feature columns")

        if text_column is not None:
            if modality != "text":
                raise InvalidDatasetError("Text column supplied, but project modality is not text")

        if text_column is None and modality == "text":
            text_column = dataset_details["text_column_guess"]
            print(f"Text column not supplied. Using best guess {text_column}")

        return api.clean_dataset(
            api_key=self._api_key,
            dataset_id=dataset_id,
            project_name=project_name,
            task_type=task_type,
            modality=modality,
            model_type=model_type,
            label_column=label_column,
            feature_columns=feature_columns if feature_columns is not None else [],
            text_column=text_column,
        )

    def wait_until_cleanset_ready(
        self, cleanset_id: str, timeout: Optional[float] = None, show_cleanset_link: bool = False
    ) -> None:
        """Blocks until a cleanset is ready or the timeout is reached.

        Args:
            cleanset_id (str): ID of cleanset to check status for.
            timeout (Optional[float], optional): timeout for polling, in seconds. Defaults to None.
            show_cleanset_link (bool, optional): whether to print a link to view the cleanset in the Cleanlab Studio web UI when the cleanset is ready. Defaults to False.

        Raises:
            TimeoutError: if cleanset is not ready by end of timeout
            CleansetError: if cleanset errored while running
        """
        clean_helpers.poll_cleanset_status(self._api_key, cleanset_id, timeout, show_cleanset_link)

    def get_latest_cleanset_id(self, project_id: str) -> str:
        """
        Gets latest cleanset ID for a project.

        Args:
            project_id: ID of project.

        Returns:
            ID of latest associated cleanset.
        """
        return api.get_latest_cleanset_id(self._api_key, project_id)

    def poll_dataset_id_for_name(self, dataset_name: str, timeout: Optional[int] = None) -> str:
        """
        Polls for dataset ID for a dataset name.

        Args:
            dataset_name: Name of dataset to get ID for.
            timeout: Optional timeout after which to stop polling for progress. If not provided, will block until dataset is ready.

        Returns
            ID of dataset.

        Raises
            TimeoutError: if dataset is not ready by end of timeout
        """
        return api.poll_dataset_id_for_name(self._api_key, dataset_name, timeout)

    def delete_project(self, project_id: str) -> None:
        """
        Deletes a project from Cleanlab Studio.

        Args:
            project_id: ID of project to delete.
        """
        api.delete_project(self._api_key, project_id)
        print(f"Successfully deleted project: {project_id}")

    def deploy_model(self, cleanset_id: str, model_name: str) -> str:
        """
        Trains and deploys a model with an improved dataset created by applying any corrections you've made to your cleanset in Cleanlab Studio.

        Args:
            cleanset_id: ID of cleanset to deploy model for.
            model_name: Name for resulting model.
        """
        return api.deploy_model(self._api_key, cleanset_id, model_name)

    def wait_until_model_ready(self, model_id: str, timeout: Optional[float] = None) -> None:
        """Blocks until a model is ready or the timeout is reached.

        Args:
            model_id (str): ID of model to check status for.
            timeout (Optional[float], optional): timeout for polling, in seconds. Defaults to None.

        Raises:
            TimeoutError: if model is not ready by end of timeout
            DeploymentError: if model errored while training
        """
        deploy_helpers.poll_deployment_status(self._api_key, model_id, timeout)

    def get_model(self, model_id: str) -> inference.Model:
        """
        Gets a model that is deployed in a Cleanlab Studio account.

        The returned model can then be used to predict labels for new data. See the documentation for the [Model](../inference#class-model) class for more on what you can do with a Model object.

        Args:
            model_id: ID of model to get. The model ID can be found in the "Model Details" tab of a model page.

        Returns:
            [Model](../inference#class-model) instance, which exposes methods to predict labels for new data.
        """
        return inference.Model(self._api_key, model_id)

    def download_pred_probs(
        self,
        cleanset_id: str,
        keep_id: bool = False,
    ) -> pd.DataFrame:
        """
        Downloads predicted probabilities for a cleanset (only for classification datasets).

        Args:
            cleanset_id (str): the ID of the cleanset for which to download the corresponding predicted class probabilities.
            keep_id (bool): whether to include the ID column in the returned DataFrame to enable easy join/merge operations with original dataset.

        Returns:
            `pd.DataFrame`: a DataFrame of probabilities of shape `N` by `M`, where `N` is the number of rows in the original dataset, and `M` is the total number of classes in the original dataset.
            Every row of the returned DataFrame corresponds to the predicted probability of each class for the corresponding row in the original dataset.
            If `keep_id` is `True`, the DataFrame will include an extra ID column that can be used for database joins/merges with
            the original dataset or downloaded Cleanlab columns.

        For image projects, a few images in the original dataset might fail to be processed due to poorly formatted data or invalid image file paths. Predicted probabilities will not be calculated for those rows.
        The rows in the original dataset that failed to be processed are marked as `True` in the `is_not_analyzed` [Cleanlab column](/guide/concepts/cleanlab_columns/#not-analyzed) of the cleanset.

        If you want to work with predicted probabilities for an image project, the recommended workflow is to download probabilities with the option `keep_id=True`, and then do a join with the original dataset on the ID column.
        Alternatively, you can follow the steps [here](/reference/python/studio#method-download_embeddings), and filter out the rows that were not analyzed. The filtered dataset will then have rows that align with the predicted probabilities DataFrame.
        """
        pred_probs: Union[npt.NDArray[np.float64], pd.DataFrame] = api.download_array(
            self._api_key, cleanset_id, "pred_probs"
        )
        if not isinstance(pred_probs, pd.DataFrame):
            pred_probs = pd.DataFrame(pred_probs)
            return pred_probs

        if not keep_id:
            id_col = api.get_id_column(self._api_key, cleanset_id)
            if id_col in pred_probs.columns:
                pred_probs = pred_probs.drop(id_col, axis=1)

        return pred_probs

    def download_embeddings(
        self,
        cleanset_id: str,
    ) -> npt.NDArray[np.float64]:
        """
        Downloads feature embeddings for a cleanset (available only for text and image projects).
        These are numeric vectors produced via neural network representations of each data point in your dataset.

        Args:
            cleanset_id (str): the ID of the cleanset from which you want to download feature embeddings.

        Returns:
            `np.NDArray[float64]`: a 2D numpy array of feature embeddings of shape `N` by `N_EMBED`, where `N` is the number of rows in the original dataset, and `N_EMBED` is the dimension of the feature embeddings. The embedding-dimension depends on which neural network is used to represent your data (Cleanlab automatically identifies the best type of neural network for your data).

        For image projects, a few images in the original dataset might fail to be processed due to poorly formatted data or invalid image file paths.
        Feature embeddings are not computed for those rows. The rows in the original dataset that failed to be processed are marked as `True` in the `is_not_analyzed` [Cleanlab column](/guide/concepts/cleanlab_columns/#not-analyzed) of the cleanset.
        If you want to work with feature embeddings for an image project, the recommended workflow is as follows:

            1. When the image project completes, download the cleaset via `studio.download_cleanlab_columns`, and check whether the `is_not_analyzed` boolean column has any `True` values.

            2. If no rows are flaged as `is_not_analyzed`, it means that all the rows were processed successfully. In this case, the rows of the feature embeddings will correspond to the rows of the original dataset, and downstream analysis can be carried out with no further preparation.

            3. If there are rows flagged as `is_not_analyzed`, the rows of the feature embeddings will correspond to the rows of the original dataset after filtering out the rows that are not analyzed.
        """
        return np.asarray(api.download_array(self._api_key, cleanset_id, "embeddings"))

    def TLM(
        self,
        quality_preset: TLMQualityPreset = "medium",
        *,
        options: Optional[trustworthy_language_model.TLMOptions] = None,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = None,
    ) -> trustworthy_language_model.TLM:
        """
        Instantiate a Trustworthy Language Model (TLM).
        For more details, see the documentation of:
        [cleanlab_studio.studio.trustworthy_language_model.TLM](../trustworthy_language_model/#class-tlm)
        """
        return trustworthy_language_model.TLM(
            self._api_key, quality_preset, options=options, timeout=timeout, verbose=verbose
        )

    def TLMHybrid(
        self,
        response_model: str = "gpt-4o",
        quality_preset: TLMQualityPreset = "medium",
        *,
        options: Optional[trustworthy_language_model.TLMOptions] = None,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = None,
    ) -> tlm_hybrid.TLMHybrid:
        """
        Instantiate a hybrid Trustworthy Language Model that uses one model for response and another for trustworthiness scoring (reduce costs/latency without reducing response quality).
        For more details, see the documentation of:
        [cleanlab_studio.utils.tlm_hybrid.TLMHybrid](../utils.tlm_hybrid/#class-tlmhybrid)
        """
        return tlm_hybrid.TLMHybrid(
            self._api_key,
            response_model,
            quality_preset,
            options=options,
            timeout=timeout,
            verbose=verbose,
        )

    def poll_cleanset_status(self, cleanset_id: str, timeout: Optional[int] = None) -> bool:
        """
        This method has been deprecated, instead use: `wait_until_cleanset_ready()`

        Repeatedly polls for cleanset status while the cleanset is being generated. Blocks until cleanset is ready, there is a cleanset error, or `timeout` is exceeded.

        Args:
            cleanset_id: ID of cleanset to check status of.
            timeout: Optional timeout after which to stop polling for progress. If not provided, will block until cleanset is ready.

        Returns:
            After cleanset is done being generated, returns `True` if cleanset is ready to use, `False` otherwise.
        """
        warnings.warn(
            "poll_cleanset_status method has been deprecated -- please use wait_for_cleanset_ready method instead.",
            DeprecationWarning,
        )

        try:
            clean_helpers.poll_cleanset_status(self._api_key, cleanset_id, timeout)
            return True

        except (TimeoutError, CleansetError):
            return False


# decorate all functions of self
#
# using variable names prepended with "_" to not create global variables that
# show up in the docs
for _name, _method in Studio.__dict__.items():
    if isinstance(_method, FunctionType):
        setattr(Studio, _name, (telemetry(track_all_frames=False))(_method))
