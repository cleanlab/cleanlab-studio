import datetime
from typing import List, Generator, Any

from cleanlab_studio.cli.classes.dataset import Dataset
import pyexcel
import pandas as pd

from cleanlab_studio.cli.types import RecordType


class ExcelDataset(Dataset):
    def read_streaming_records(self) -> Generator[RecordType, None, None]:
        for r in pyexcel.iget_records(file_name=self.filepath):
            yield self._preprocess_record(r)

    def read_streaming_values(self) -> Generator[List[Any], None, None]:
        stream = pyexcel.iget_array(file_name=self.filepath)
        next(stream)  # skip header row
        for r in stream:
            yield self._preprocess_values(r)

    def read_file_as_dataframe(self) -> pd.DataFrame:
        return pd.read_excel(self.filepath, keep_default_na=True)

    def get_columns(self) -> List[str]:
        stream = pyexcel.iget_array(file_name=self.filepath)
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
        if isinstance(record_value, (datetime.time, datetime.datetime, pd.Timestamp)):
            return record_value.isoformat()

        return record_value
