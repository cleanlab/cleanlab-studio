from typing import Tuple, Optional

from cleanlab_studio.dataset import (
    RowWarningsType,
    FeatureType,
    DataType,
)
from cleanlab_studio.types import RecordType
from tests.row_processing.utils import process_record_with_fields


class TestDatetime:
    @staticmethod
    def process_datetime(
        record: RecordType, data_type: str
    ) -> Tuple[Optional[RecordType], Optional[str], Optional[RowWarningsType]]:
        return process_record_with_fields(
            record=record,
            fields={
                "timestamp": dict(data_type=data_type, feature_type=FeatureType.datetime.value)
            },
        )

    def test_float_input(self) -> None:
        row, row_id, warnings = self.process_datetime(
            {"timestamp": 180.0}, data_type=DataType.float.value
        )
        assert row["timestamp"] == 180.0

    def test_nan_input(self) -> None:
        row, row_id, warnings = self.process_datetime(
            {"timestamp": float("nan")}, data_type=DataType.float.value
        )
        assert row["timestamp"] is None

    def test_null_input(self) -> None:
        row, row_id, warnings = self.process_datetime(
            {"timestamp": None}, data_type=DataType.float.value
        )
        assert row["timestamp"] is None

    def test_integer_input(self) -> None:
        row, row_id, warnings = self.process_datetime(
            {"timestamp": 189000000}, data_type=DataType.integer.value
        )
        assert row["timestamp"] == 189000000

    def test_valid_string_input(self) -> None:
        row, row_id, warnings = self.process_datetime(
            record={"timestamp": "2015-02-24 11:35:52 -0800"}, data_type=DataType.string.value
        )
        assert row["timestamp"] == "2015-02-24 11:35:52 -0800"

    def test_valid_string_input_2(self) -> None:
        row, row_id, warnings = self.process_datetime(
            record={"timestamp": "2015-02-24"}, data_type=DataType.string.value
        )
        assert row["timestamp"] == "2015-02-24"

    def test_valid_string_input_3(self) -> None:
        row, row_id, warnings = self.process_datetime(
            record={"timestamp": "24 Feb 2015"}, data_type=DataType.string.value
        )
        assert row["timestamp"] == "24 Feb 2015"

    def test_valid_string_input_4(self) -> None:
        row, row_id, warnings = self.process_datetime(
            record={"timestamp": "Feb 24 2015"}, data_type=DataType.string.value
        )
        assert row["timestamp"] == "Feb 24 2015"

    def test_invalid_string_input(self) -> None:
        row, row_id, warnings = self.process_datetime(
            record={"timestamp": "abc"}, data_type=DataType.string.value
        )
        assert row["timestamp"] is None

    ## Data type and input type mismatch
    def test_integer_string_input_with_integer_data_type(self) -> None:
        row, row_id, warnings = self.process_datetime(
            {"timestamp": "189000000"}, data_type=DataType.integer.value
        )
        assert row["timestamp"] == 189000000

    def test_invalid_string_input_with_integer_data_type(self) -> None:
        row, row_id, warnings = self.process_datetime(
            {"timestamp": "189000000a"}, data_type=DataType.integer.value
        )
        assert row["timestamp"] is None

    def test_float_string_input_with_integer_data_type(self) -> None:
        row, row_id, warnings = self.process_datetime(
            {"timestamp": "189000000.0"}, data_type=DataType.integer.value
        )
        assert row["timestamp"] == 189000000  # TODO

    def test_float_input_with_integer_data_type(self) -> None:
        row, row_id, warnings = self.process_datetime(
            {"timestamp": 189000000.0}, data_type=DataType.integer.value
        )
        assert row["timestamp"] == 189000000

    def test_integer_string_input_with_float_data_type(self) -> None:
        row, row_id, warnings = self.process_datetime(
            {"timestamp": "189000000"}, data_type=DataType.float.value
        )
        assert row["timestamp"] == 189000000.0

    def test_invalid_string_input_with_float_data_type(self) -> None:
        row, row_id, warnings = self.process_datetime(
            {"timestamp": "189000000a"}, data_type=DataType.float.value
        )
        assert row["timestamp"] is None

    def test_integer_input_with_string_date_type(self) -> None:
        row, row_id, warnings = self.process_datetime(
            {"timestamp": 189000000}, data_type=DataType.string.value
        )
        assert row["timestamp"] is None

    def test_float_input_with_string_date_type(self) -> None:
        row, row_id, warnings = self.process_datetime(
            {"timestamp": 189000000.0}, data_type=DataType.string.value
        )
        assert row["timestamp"] is None
