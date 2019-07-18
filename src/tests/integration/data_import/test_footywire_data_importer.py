import os
from unittest import TestCase
from datetime import date, datetime

import pandas as pd

from machine_learning.data_import import FootywireDataImporter
from machine_learning.settings import MELBOURNE_TIMEZONE

START_OF_LAST_YEAR = f"{date.today().year - 1}-01-01"
TEST_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../fixtures")
)


class TestFootywireDataImporter(TestCase):
    def setUp(self):
        self.data_reader = FootywireDataImporter(json_dir=TEST_DATA_DIR, verbose=0)

    def test_get_betting_odds(self):
        with self.subTest("when fetch_data is True"):
            data_frame = self.data_reader.get_betting_odds(
                start_date="2014-01-01", end_date="2015-12-31", fetch_data=True
            )

            self.assertIsInstance(data_frame, pd.DataFrame)

            seasons = data_frame["season"].drop_duplicates()
            self.assertEqual(len(seasons), 2)
            self.assertEqual(seasons.iloc[0], 2014)

            self.assertEqual(data_frame["date"].dtype, "datetime64[ns, UTC+11:00]")
            date_years = data_frame["date"].dt.year.drop_duplicates()
            self.assertEqual(len(date_years), 2)
            self.assertEqual(date_years.iloc[0], 2014)

            with self.subTest("and default end date is used"):
                data_frame = self.data_reader.get_betting_odds(
                    start_date=START_OF_LAST_YEAR, fetch_data=True
                )

                right_now = datetime.now(  # pylint: disable=unused-variable
                    tz=MELBOURNE_TIMEZONE
                )
                future_betting_data = data_frame.query("date > @right_now")

                self.assertTrue(any(future_betting_data))

        with self.subTest("when fetch_data is False"):
            data_frame = self.data_reader.get_betting_odds(fetch_data=False)

            self.assertIsInstance(data_frame, pd.DataFrame)
            self.assertFalse(data_frame.empty)
            self.assertEqual(data_frame["date"].dtype, "datetime64[ns, UTC+11:00]")

            with self.subTest("and year_range is specified"):
                data_frame = self.data_reader.get_betting_odds(
                    start_date="2018-01-01", end_date="2019-12-31", fetch_data=False
                )

                self.assertIsInstance(data_frame, pd.DataFrame)
                seasons = data_frame["season"].drop_duplicates()
                self.assertEqual(len(seasons), 1)
                self.assertEqual(seasons.iloc[0], 2018)
