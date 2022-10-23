from typing import Tuple, Optional

from cleanlab_studio.cli.dataset import (
    RowWarningsType,
    DataType,
    FeatureType,
)
from cleanlab_studio.cli.types import RecordType
from tests.row_processing.utils import process_record_with_fields


class TestIntegerDataType:
    @staticmethod
    def process_with_integer_numeric(
        record: RecordType,
    ) -> Tuple[Optional[RecordType], Optional[str], Optional[RowWarningsType]]:
        return process_record_with_fields(
            record=record,
            fields={
                "height": dict(
                    data_type=DataType.integer.value, feature_type=FeatureType.numeric.value
                )
            },
        )

    def test_float_integer_input(self) -> None:
        row, row_id, warnings = self.process_with_integer_numeric({"height": 180.0})
        assert row["height"] == 180

    def test_float_non_integer_input(self) -> None:
        row, row_id, warnings = self.process_with_integer_numeric({"height": 180.5})
        assert row["height"] is None

    def test_string_integer_input(self) -> None:
        row, row_id, warnings = self.process_with_integer_numeric({"height": "180"})
        assert row["height"] == 180

    def test_string_non_integer_input(self) -> None:
        row, row_id, warnings = self.process_with_integer_numeric({"height": "180a"})
        assert row["height"] is None

    def test_float_string_input(self) -> None:
        row, row_id, warnings = self.process_with_integer_numeric({"height": "180.0"})
        assert row["height"] == 180

    def test_float_string_input_2(self) -> None:
        row, row_id, warnings = self.process_with_integer_numeric({"height": "180.5"})
        assert row["height"] is None

    def test_scientific_input(self) -> None:
        row, row_id, warnings = self.process_with_integer_numeric({"height": "1e9"})
        assert row["height"] == 1000000000

    def test_boolean_input(self) -> None:
        row, row_id, warnings = self.process_with_integer_numeric({"height": True})
        assert row["height"] == 1

    def test_null_input(self) -> None:
        row, row_id, warnings = self.process_with_integer_numeric({"height": None})
        assert row["height"] is None

    def test_nan_input(self) -> None:
        row, row_id, warnings = self.process_with_integer_numeric({"height": float("nan")})
        assert row["height"] is None
