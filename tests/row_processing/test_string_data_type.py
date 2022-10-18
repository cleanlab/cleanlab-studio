from typing import Tuple, Optional

from cleanlab_cli.dataset import RowWarningsType, DataType, FeatureType
from cleanlab_cli.types import RecordType
from tests.row_processing.utils import process_record_with_fields


class TestStringDataType:
    @staticmethod
    def process_with_string_data_type(
        record: RecordType,
    ) -> Tuple[Optional[RecordType], Optional[str], Optional[RowWarningsType]]:
        return process_record_with_fields(
            record=record,
            fields={
                "height": dict(data_type=DataType.string.value, feature_type=FeatureType.text.value)
            },
        )

    def test_float_input(self) -> None:
        row, row_id, warnings = self.process_with_string_data_type({"height": 180.5})
        assert row["height"] == "180.5"

    def test_string_input(self) -> None:
        row, row_id, warnings = self.process_with_string_data_type({"height": "180cm"})
        assert row["height"] == "180cm"

    def test_integer_input(self) -> None:
        row, row_id, warnings = self.process_with_string_data_type({"height": 180})
        assert row["height"] == "180"

    def test_boolean_input(self) -> None:
        row, row_id, warnings = self.process_with_string_data_type({"height": True})
        assert row["height"] == "True"

    def test_scientific_input(self) -> None:
        row, row_id, warnings = self.process_with_string_data_type({"height": 1e9})
        assert row["height"] == "1000000000.0"

    def test_null_input(self) -> None:
        row, row_id, warnings = self.process_with_string_data_type({"height": None})
        assert row["height"] is None

    def test_nan_input(self) -> None:
        row, row_id, warnings = self.process_with_string_data_type({"height": float("nan")})
        assert row["height"] is None
