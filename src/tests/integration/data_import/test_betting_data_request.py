# pylint: disable=missing-module-docstring, missing-function-docstring
# pylint: disable=missing-class-docstring

import os
from unittest import TestCase, skip
from unittest.mock import patch, MagicMock
from datetime import date, timedelta

from betamax import Betamax
from requests import Session

from augury.data_import.betting_data import fetch_betting_data
from augury.settings import CASSETTE_LIBRARY_DIR

SEPT = 9
MAR = 3
FIFTEENTH = 15
THIRTY_FIRST = 31
AFL_DATA_SERVICE = os.getenv("AFL_DATA_SERVICE", default="")
AFL_DATA_SERVICE_TOKEN = os.getenv("AFL_DATA_SERVICE_TOKEN", default="")
ENV_VARS = os.environ.copy()
DATA_IMPORT_PATH = "augury.data_import"

with Betamax.configure() as config:
    config.cassette_library_dir = CASSETTE_LIBRARY_DIR
    config.define_cassette_placeholder("<AFL_DATA_TOKEN>", AFL_DATA_SERVICE_TOKEN)
    config.define_cassette_placeholder("<AFL_DATA_URL>", AFL_DATA_SERVICE)


class TestBettingData(TestCase):
    def setUp(self):
        today = date.today()
        # Season start and end are approximate, but defined to be safely after the
        # usual start and before the usual end
        end_of_previous_season = date(today.year - 1, SEPT, FIFTEENTH)
        start_of_this_season = date(today.year, MAR, THIRTY_FIRST)
        end_of_this_season = date(today.year, SEPT, FIFTEENTH)

        a_month = timedelta(days=30)
        a_month_ago = today - a_month

        if today >= start_of_this_season and a_month_ago < end_of_this_season:
            self.start_date = str(a_month_ago)
        elif today < start_of_this_season:
            self.start_date = str(end_of_previous_season - a_month)
        else:
            self.start_date = str(end_of_this_season - a_month)

        self.end_date = str(today)

    @skip("Data is blank because there's no AFL season because pandemic")
    def test_fetch_betting_data(self):
        data = fetch_betting_data(
            start_date=self.start_date, end_date=self.end_date, verbose=0
        )

        self.assertIsInstance(data, list)
        self.assertIsInstance(data[0], dict)
        self.assertTrue(any(data))

        dates = {datum["date"] for datum in data}
        self.assertLessEqual(self.start_date, min(dates))
        self.assertGreaterEqual(self.end_date, max(dates))


class TestBettingDataProd(TestCase):
    def setUp(self):
        self.session = Session()
        self.start_date = "2012-01-01"
        self.end_date = "2013-12-31"

    @skip("Data is sometimes blank and the backup scraper isn't working in production")
    @patch.dict(os.environ, {**ENV_VARS, **{"PYTHON_ENV": "production"}}, clear=True)
    @patch(f"{DATA_IMPORT_PATH}.betting_data.json.dump", MagicMock())
    def test_fetch_betting_data(self):
        with Betamax(self.session).use_cassette("betting_data"):
            with patch(f"{DATA_IMPORT_PATH}.base_data.requests.get", self.session.get):
                data = fetch_betting_data(
                    start_date=self.start_date, end_date=self.end_date, verbose=0
                )
                self.assertIsInstance(data, list)
                self.assertIsInstance(data[0], dict)
                self.assertTrue(any(data))

                dates = {datum["date"] for datum in data}
                self.assertLessEqual(self.start_date, min(dates))
                self.assertGreaterEqual(self.end_date, max(dates))
