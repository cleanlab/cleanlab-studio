from typing import List, Optional

from cleanlab_studio.internal.api import actions_api
from cleanlab_studio.internal import clean_helpers
from cleanlab_studio.internal.types import JSONDict


class Cleanset:
    """A Cleanset corresponds to the cleaned version of your dataset at the current stage of a Project.

    It includes:
    - User data modifications that have been made thus far
    - Cleanlab metadata columns generated during the analysis
    """

    def __init__(self, api_key: str, cleanset_id: str, timeout: Optional[int]):
        """Initializes a Cleanset object with the given API key and cleanset ID.

        If cleanset is not ready, blocks until timeout is reached or cleanset is ready.
        """
        self._api_key = api_key
        self._cleanset_id = cleanset_id

        # block until timed out or cleanset is ready
        clean_helpers.poll_cleanset_status(self._api_key, cleanset_id, timeout)

    # TODO: add
    # - download_cleanlab_columns
    # - apply_corrections
    # - download_pred_probs
    # - download_embeddings
    # - wait until ready / poll cleanset status (class methods)

    def read_row(
        self,
        row_id: clean_helpers.RowId,
        dataset_columns: Optional[List[str]] = None,
        cleanlab_columns: Optional[List[str]] = None,
    ) -> JSONDict:
        """Reads a row from the cleanset, returning a JSON dict with the row's data.
        By default, this will return all dataset and all cleanlab columns for a row.

        Args:
            row_id (clean_helpers.RowId): row ID to get data for
            dataset_columns (Optional[List[str]], optional): list of dataset columns to include in returned row, defaults to None.
            cleanlab_columns (Optional[List[str]], optional): list of cleanlab columns to include in returned row, defaults to None.

        Returns:
            JSONDict: dictionary containing key-value pairs for row in cleanset

        Raises:
            RowIdTypeError: if row ID type does not match type in cleanset
            RowNotFoundError: if row ID is not found in cleanset
            ColumnNotFoundError: if column (provided in dataset or cleanlab columns) is not found in cleanset
            TaskTypeError: if this operation is not supported for the task type of the cleanset
        """
        return actions_api.read_row(
            self._api_key,
            self._cleanset_id,
            row_id,
            dataset_columns,
            cleanlab_columns,
        )

    def get_possible_labels(self) -> List[clean_helpers.Label]:
        """Gets list of possible labels for a project.

        When updating labels in a cleanset, only labels that are present in the initially uploaded dataset can be used.

        Returns:
            List[clean_helpers.Label]: list of candidate labels for a cleanset

        Raises:
            TaskTypeError: if this operation is not supported for the task type of the cleanset
        """
        return clean_helpers.get_possible_labels(
            self._api_key,
            self._cleanset_id,
        )

    def exclude(
        self,
        row_id: clean_helpers.RowId,
    ) -> None:
        """Excludes row from cleanset.

        This omits a row from the cleanset.
        This operation is used when you wish to throw away a data point.
        One example of this is if you have a data point that is a duplicate of another data point.

        Args:
            row_id (clean_helpers.RowId): row ID to exclude

        Raises:
            RowIdTypeError: if row ID type does not match type in cleanset
            RowNotFoundError: if row ID is not found in cleanset
            TaskTypeError: if this operation is not supported for the task type of the cleanset
        """
        clean_helpers.exclude(
            self._api_key,
            self._cleanset_id,
            row_id,
        )

    def keep(
        self,
        row_id: clean_helpers.RowId,
    ) -> None:
        """Keeps given label for row in cleanset.

        This is used to mark a row as correctly labeled.

        Args:
            row_id (clean_helpers.RowId): row ID to keep label for

        Raises:
            RowIdTypeError: if row ID type does not match type in cleanset
            RowNotFoundError: if row ID is not found in cleanset
            TaskTypeError: if this operation is not supported for the task type of the cleanset
        """
        clean_helpers.keep(
            self._api_key,
            self._cleanset_id,
            row_id,
        )

    def relabel(
        self,
        row_id: clean_helpers.RowId,
        label: clean_helpers.Label,
    ) -> None:
        """Keeps given label for row in cleanset.

        This is used to mark a row as correctly labeled.

        Args:
            row_id (clean_helpers.RowId): row ID to relabel
            label (clean_helpers.Label): label value to relabel row with

        Raises:
            RowIdTypeError: if row ID type does not match type in cleanset
            RowNotFoundError: if row ID is not found in cleanset
            LabelTypeError: if label type does not match type in cleanset
            LabelValueError: if label value is not in the valid set of labels for the cleanset
            TaskTypeError: if this operation is not supported for the task type of the cleanset
        """
        clean_helpers.relabel(
            self._api_key,
            self._cleanset_id,
            row_id,
            label,
        )
