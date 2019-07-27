from unittest import TestCase
from unittest.mock import MagicMock, patch

from machine_learning.io import JSONRemoteDataSet


class TestJSONRemoteDataSet(TestCase):
    def setUp(self):
        self.start_date = "2018-01-01"
        self.end_date = "2018-12-31"
        self.data_source = MagicMock()
        self.data_set = JSONRemoteDataSet(
            start_date=self.start_date,
            end_date=self.end_date,
            data_source=self.data_source,
        )

    def test_load(self):
        self.data_set.load()

        self.data_source.assert_called_with(
            start_date=self.start_date, end_date=self.end_date
        )

        with self.subTest("with string path to data_source function"):
            data_source_path = (
                "machine_learning.data_import.betting_data.fetch_betting_data"
            )

            with patch(data_source_path):
                data_set = JSONRemoteDataSet(
                    start_date=self.start_date,
                    end_date=self.end_date,
                    data_source=data_source_path,
                )

                data_set.load()

                data_set.data_source.assert_called_with(
                    start_date=self.start_date, end_date=self.end_date
                )

    def test_save(self):
        self.data_set.save({})
