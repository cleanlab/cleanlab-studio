from typing import Tuple, Optional, Dict

from cleanlab_studio.version import SCHEMA_VERSION
from cleanlab_studio.cli.classes.dataset import Dataset
from cleanlab_studio.cli.dataset import validate_and_process_record, Schema, RowWarningsType
from cleanlab_studio.cli.dataset.schema_types import SchemaMetadata
from cleanlab_studio.cli.types import RecordType

ID_COLUMN = "id"


def initialize_schema_from_fields(fields: Dict[str, Dict[str, str]]) -> Schema:
    schema_metadata = dict(
        id_column=ID_COLUMN, modality="tabular", name="dummy_file", filepath_column=None
    )
    return Schema.create(metadata=schema_metadata, fields=fields, version=SCHEMA_VERSION)


def process_record_with_fields(
    record: RecordType, fields: Dict[str, Dict[str, str]]
) -> Tuple[Optional[RecordType], Optional[str], Optional[RowWarningsType]]:
    record[ID_COLUMN] = "id"
    schema = initialize_schema_from_fields(fields)
    row, row_id, warnings = validate_and_process_record(
        schema=schema,
        record=record,
        seen_ids=set(),
        existing_ids=set(),
    )
    return row, row_id, warnings
