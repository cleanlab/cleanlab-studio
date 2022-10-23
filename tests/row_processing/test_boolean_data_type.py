from typing import Tuple, Optional

from cleanlab_studio.dataset import (
    RowWarningsType,
    DataType,
    FeatureType,
)
from cleanlab_studio.types import RecordType
from tests.row_processing.utils import process_record_with_fields


class TestBooleanDataType:
    @staticmethod
    def process_with_boolean(
        record: RecordType,
    ) -> Tuple[Optional[RecordType], Optional[str], Optional[RowWarningsType]]:
        return process_record_with_fields(
            record=record,
            fields={
                "x": dict(data_type=DataType.boolean.value, feature_type=FeatureType.boolean.value)
            },
        )

    def test_boolean_input(self) -> None:
        row, row_id, warnings = self.process_with_boolean({"x": True})
        assert row["x"] is True

    def test_yes_input(self) -> None:
        row, row_id, warnings = self.process_with_boolean({"x": "yes"})
        assert row["x"] is True

    def test_no_input(self) -> None:
        row, row_id, warnings = self.process_with_boolean({"x": "no"})
        assert row["x"] is False

    def test_true_string_input(self) -> None:
        row, row_id, warnings = self.process_with_boolean({"x": "true"})
        assert row["x"] is True

    def test_false_string_input(self) -> None:
        row, row_id, warnings = self.process_with_boolean({"x": "false"})
        assert row["x"] is False

    def test_t_input(self) -> None:
        row, row_id, warnings = self.process_with_boolean({"x": "t"})
        assert row["x"] is True

    def test_f_input(self) -> None:
        row, row_id, warnings = self.process_with_boolean({"x": "f"})
        assert row["x"] is False

    def test_y_input(self) -> None:
        row, row_id, warnings = self.process_with_boolean({"x": "y"})
        assert row["x"] is True

    def test_n_input(self) -> None:
        row, row_id, warnings = self.process_with_boolean({"x": "n"})
        assert row["x"] is False

    def test_1_string_input(self) -> None:
        row, row_id, warnings = self.process_with_boolean({"x": "1"})
        assert row["x"] is True

    def test_0_string_input(self) -> None:
        row, row_id, warnings = self.process_with_boolean({"x": "0"})
        assert row["x"] is False

    def test_1_integer_input(self) -> None:
        row, row_id, warnings = self.process_with_boolean({"x": 1})
        assert row["x"] is True

    def test_0_integer_input(self) -> None:
        row, row_id, warnings = self.process_with_boolean({"x": 0})
        assert row["x"] is False

    def test_invalid_integer_input(self) -> None:
        row, row_id, warnings = self.process_with_boolean({"x": 100})
        assert row["x"] is None

    def test_invalid_string_input(self) -> None:
        row, row_id, warnings = self.process_with_boolean({"x": "ok"})
        assert row["x"] is None

    def test_1_float_input(self) -> None:
        row, row_id, warnings = self.process_with_boolean({"x": 1.0})
        assert row["x"] is True

    def test_0_float_input(self) -> None:
        row, row_id, warnings = self.process_with_boolean({"x": 0.0})
        assert row["x"] is False

    def test_null_input(self) -> None:
        row, row_id, warnings = self.process_with_boolean({"x": None})
        assert row["x"] is None

    def test_nan_input(self) -> None:
        row, row_id, warnings = self.process_with_boolean({"x": float("nan")})
        assert row["x"] is None
