# pylint: disable=missing-module-docstring, missing-function-docstring
# pylint: disable=missing-class-docstring

import os
from unittest import TestCase
from unittest.mock import patch, MagicMock
from datetime import date, timedelta

from betamax import Betamax
from requests import Session

from augury.data_import.match_data import fetch_match_data, fetch_fixture_data
from augury.settings import CASSETTE_LIBRARY_DIR

SEPT = 9
MAR = 3
FIFTEENTH = 15
THIRTY_FIRST = 31
AFL_DATA_SERVICE = os.getenv("AFL_DATA_SERVICE", default="")
GCR_TOKEN = os.getenv("GCR_TOKEN", default="")
ENV_VARS = os.environ.copy()
DATA_IMPORT_PATH = "augury.data_import"

with Betamax.configure() as config:
    config.cassette_library_dir = CASSETTE_LIBRARY_DIR
    config.define_cassette_placeholder("<AFL_DATA_TOKEN>", GCR_TOKEN)
    config.define_cassette_placeholder("<AFL_DATA_URL>", AFL_DATA_SERVICE)


class TestMatchData(TestCase):
    def setUp(self):
        today = date.today()
        # Season start and end are approximate, but defined to be safely after the
        # usual start and before the usual end
        end_of_previous_season = date(today.year - 1, SEPT, FIFTEENTH)
        start_of_this_season = date(today.year, MAR, THIRTY_FIRST)
        end_of_this_season = date(today.year, SEPT, FIFTEENTH)

        a_year = timedelta(days=365)
        a_year_ago = today - a_year

        if today > start_of_this_season and a_year_ago < end_of_this_season:
            self.start_date = str(a_year_ago)
        elif today < start_of_this_season:
            self.start_date = str(end_of_previous_season - a_year)
        else:
            self.start_date = str(end_of_this_season - a_year)

        self.end_date = str(today)

    def test_fetch_match_data(self):
        data = fetch_match_data(
            start_date=self.start_date, end_date=self.end_date, verbose=0
        )

        self.assertIsInstance(data, list)
        self.assertIsInstance(data[0], dict)
        self.assertTrue(any(data))

        dates = {datum["date"] for datum in data}
        self.assertLessEqual(self.start_date, min(dates))
        self.assertGreaterEqual(self.end_date, max(dates))

    def test_fetch_fixture_data(self):
        # Fixture data doesn't go very far back and is mostly for getting upcoming
        # match data during the season, so these dates are better examples
        # of actual usage
        start_date = "2016-05-01"
        end_date = "2016-08-31"

        data = fetch_fixture_data(start_date=start_date, end_date=end_date, verbose=0)

        self.assertIsInstance(data, list)
        self.assertIsInstance(data[0], dict)
        self.assertTrue(any(data))

        dates = {datum["date"] for datum in data}
        self.assertLessEqual(start_date, min(dates))
        self.assertGreaterEqual(end_date, max(dates))


@patch.dict(os.environ, {**ENV_VARS, **{"PYTHON_ENV": "production"}}, clear=True)
@patch(f"{DATA_IMPORT_PATH}.match_data.json.dump", MagicMock())
class TestMatchDataProd(TestCase):
    def setUp(self):
        self.session = Session()
        self.start_date = "2012-01-01"
        self.end_date = "2013-12-31"

    def test_fetch_match_data(self):
        with Betamax(self.session).use_cassette("match_data"):
            with patch(f"{DATA_IMPORT_PATH}.base_data.requests.get", self.session.get):
                data = fetch_match_data(
                    start_date=self.start_date, end_date=self.end_date, verbose=0
                )
                self.assertIsInstance(data, list)
                self.assertIsInstance(data[0], dict)
                self.assertTrue(any(data))

                dates = {datum["date"] for datum in data}
                self.assertLessEqual(self.start_date, min(dates))
                self.assertGreaterEqual(self.end_date, max(dates))

    def test_fetch_fixture_data(self):
        # Fixture data doesn't go very far back and is mostly for getting upcoming
        # match data during the season, so these dates are better examples
        # of actual usage
        start_date = "2016-05-01"
        end_date = "2016-08-31"

        with Betamax(self.session).use_cassette("fixture_data"):
            with patch(f"{DATA_IMPORT_PATH}.base_data.requests.get", self.session.get):
                data = fetch_fixture_data(
                    start_date=start_date, end_date=end_date, verbose=0
                )
                self.assertIsInstance(data, list)
                self.assertIsInstance(data[0], dict)
                self.assertTrue(any(data))

                dates = {datum["date"] for datum in data}
                self.assertLessEqual(start_date, min(dates))
                self.assertGreaterEqual(end_date, max(dates))
