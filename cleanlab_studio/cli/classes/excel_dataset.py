import datetime
from typing import List, Generator, Any, Dict, IO

from cleanlab_studio.cli.classes.dataset import Dataset
import pyexcel
import pandas as pd

from cleanlab_studio.cli.types import RecordType


class ExcelDataset(Dataset[IO[bytes]]):
    READ_ARGS: Dict[str, str] = {"mode": "rb"}

    def __init__(self, *args: Any, file_type: str, **kwargs: Any):
        self._file_type = file_type
        super().__init__(*args, **kwargs)

    def read_streaming_records(self) -> Generator[RecordType, None, None]:
        with self.fileobj() as f:
            for r in pyexcel.iget_records(file_stream=f, file_type=self._file_type):
                yield self._preprocess_record(r)

    def read_streaming_values(self) -> Generator[List[Any], None, None]:
        with self.fileobj() as f:
            stream = pyexcel.iget_array(file_stream=f, file_type=self._file_type)
            next(stream)  # skip header row
            for r in stream:
                yield self._preprocess_values(r)

    def read_file_as_dataframe(self) -> pd.DataFrame:
        with self.fileobj() as f:
            return pd.read_excel(f, keep_default_na=True)

    def get_columns(self) -> List[str]:
        with self.fileobj() as f:
            stream = pyexcel.iget_array(file_stream=f, file_type=self._file_type)

        return [str(col) for col in next(stream)]

    def _preprocess_record(self, record: RecordType) -> RecordType:
        """Preprocesses record.

        :param record: record to preprocess
        :return: preprocess record
        """
        return {
            record_key: self._preprocess_value(record_value)
            for record_key, record_value in record.items()
        }

    def _preprocess_values(self, values: List[Any]) -> List[Any]:
        """Preprocesses values.

        :param values: values to preprocess
        :return: preprocess values
        """
        return [self._preprocess_value(value) for value in values]

    def _preprocess_value(self, record_value: Any) -> Any:
        """Preprocesses record value.
        Operations performed:
            - Cast datetimes to strings

        :param record_value: record value to preprocess
        :return: preprocessed record value
        """
        if isinstance(
            record_value, (datetime.date, datetime.time, datetime.datetime, pd.Timestamp)
        ):
            return record_value.isoformat()

        return record_value
