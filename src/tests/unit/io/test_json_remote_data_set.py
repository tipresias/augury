# pylint: disable=missing-module-docstring, missing-function-docstring
# pylint: disable=missing-class-docstring

from unittest import TestCase
from unittest.mock import MagicMock, patch

from augury.io.json_remote_data_set import JSONRemoteDataSet, DATE_RANGE_TYPE


class TestJSONRemoteDataSet(TestCase):
    def setUp(self):
        self.date_range_type = "past_rounds"
        self.data_source = MagicMock()
        self.data_set = JSONRemoteDataSet(
            data_source=self.data_source, date_range_type=self.date_range_type
        )

    def test_load(self):
        self.data_set.load()

        self.data_source.assert_called_with(**DATE_RANGE_TYPE[self.date_range_type])

        with self.subTest("with string path to data_source function"):
            data_source_path = "augury.data_import.betting_data.fetch_betting_data"

            with patch(data_source_path):
                data_set = JSONRemoteDataSet(
                    date_range_type=self.date_range_type, data_source=data_source_path
                )

                data_set.load()

                data_set.data_source.assert_called_with(
                    **DATE_RANGE_TYPE[self.date_range_type]
                )

    def test_save(self):
        self.data_set.save({})
