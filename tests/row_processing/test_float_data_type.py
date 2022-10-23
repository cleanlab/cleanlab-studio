from typing import Tuple, Optional

from cleanlab_studio.dataset import (
    RowWarningsType,
    DataType,
    FeatureType,
)
from cleanlab_studio.types import RecordType
from tests.row_processing.utils import process_record_with_fields


class TestFloatDataType:
    @staticmethod
    def process_with_float(
        record: RecordType,
    ) -> Tuple[Optional[RecordType], Optional[str], Optional[RowWarningsType]]:
        return process_record_with_fields(
            record=record,
            fields={
                "x": dict(data_type=DataType.float.value, feature_type=FeatureType.numeric.value)
            },
        )

    # def test_boolean_input(self) -> None:
    #     row, row_id, warnings = self.process_with_float({"x": True})
    #     assert row["x"] is None

    def test_float_string_input(self) -> None:
        row, row_id, warnings = self.process_with_float({"x": "180.0"})
        assert row["x"] == 180.0

    def test_integer_string_input(self) -> None:
        row, row_id, warnings = self.process_with_float({"x": "180"})
        assert row["x"] == 180.0

    def test_invalid_string_input(self) -> None:
        row, row_id, warnings = self.process_with_float({"x": "cc982"})
        assert row["x"] is None

    def test_percentage_string_input(self) -> None:
        row, row_id, warnings = self.process_with_float({"x": "180%"})
        assert row["x"] == 180.0

    def test_integer_input(self) -> None:
        row, row_id, warnings = self.process_with_float({"x": 180})
        assert row["x"] == 180

    def test_float_input(self) -> None:
        row, row_id, warnings = self.process_with_float({"x": 180.0})
        assert row["x"] == 180.0

    def test_nan_input(self) -> None:
        row, row_id, warnings = self.process_with_float({"x": float("nan")})
        assert row["x"] is None

    def test_null_input(self) -> None:
        row, row_id, warnings = self.process_with_float({"x": None})
        assert row["x"] is None
