"""
Methods for interfacing with Enrichment Projects.

**This module is not meant to be imported and used directly.** Instead, use [`Studio.get_enrichment_project()`](../studio/#method-get_enrichment_project) to instantiate an [EnrichmentProject](#class-enrichmentproject) object.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional

from cleanlab_studio.internal.api import api


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
        target_column_in_dataset: str,
        created_at_string: Optional[str] = None,
    ) -> None:
        """Initialize an EnrichmentProject.

        **Objects of this class are not meant to be constructed directly.** Instead, use [`Studio.get_enrichment_project()`](../studio/#method-get_enrichment_project).
        """
        self._api_key = api_key
        self._id = id
        self._name = name
        self._created_at_string = created_at_string
        self.target_column_in_dataset = target_column_in_dataset

    def _get_project_dict(self):
        return dict(api.get_enrichment_project(api_key=self._api_key, project_id=self._id))

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def created_at(self) -> datetime:
        if self._created_at_string is None:
            self._created_at_string = self._get_project_dict()["created_at"]

        return _response_timestamp_to_datetime(self._created_at_string)

    @property
    def updated_at(self) -> datetime:
        updated_at = self._get_project_dict()["updated_at"]
        return _response_timestamp_to_datetime(updated_at)
