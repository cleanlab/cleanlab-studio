"""
Methods for interfacing with Enrichment Projects.

**This module is not meant to be imported and used directly.** Instead, use [`Studio.get_enrichment_project()`](../studio/#method-get_enrichment_project) to instantiate an [EnrichmentProject](#class-enrichmentproject) object.
"""

from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, Optional, Union

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
        created_at: Optional[Union[str, datetime]] = None,
    ) -> None:
        """Initialize an EnrichmentProject.

        **Objects of this class are not meant to be constructed directly.**
        Instead, use [`Studio.get_enrichment_project()`](../studio/#method-get_enrichment_project).
        """
        self._api_key = api_key
        self._id = id
        self._name = name
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
