import os
from unittest import TestCase
from unittest.mock import patch, MagicMock
from datetime import date, timedelta

from betamax import Betamax
from requests import Session

from machine_learning.data_import.betting_data import fetch_betting_data
from machine_learning.settings import CASSETTE_LIBRARY_DIR

SEPT = 9
MAR = 3
FIFTEENTH = 15
THIRTY_FIRST = 31
AFL_DATA_SERVICE = os.getenv("AFL_DATA_SERVICE", default="")
GCR_TOKEN = os.getenv("GCR_TOKEN", default="")
ENV_VARS = os.environ.copy()
DATA_IMPORT_PATH = "machine_learning.data_import"

with Betamax.configure() as config:
    config.cassette_library_dir = CASSETTE_LIBRARY_DIR
    config.define_cassette_placeholder("<AFL_DATA_TOKEN>", GCR_TOKEN)
    config.define_cassette_placeholder("<AFL_DATA_URL>", AFL_DATA_SERVICE)


# class TestBettingData(TestCase):
#     def setUp(self):
#         today = date.today()
#         # Season start and end are approximate, but defined to be safely after the
#         # usual start and before the usual end
#         end_of_previous_season = date(today.year - 1, SEPT, FIFTEENTH)
#         start_of_this_season = date(today.year, MAR, THIRTY_FIRST)
#         end_of_this_season = date(today.year, SEPT, FIFTEENTH)

#         a_month = timedelta(days=30)
#         a_month_ago = today - a_month

#         if today > start_of_this_season and a_month_ago < end_of_this_season:
#             self.start_date = str(a_month_ago)
#         elif today < start_of_this_season:
#             self.start_date = str(end_of_previous_season - a_month)
#         else:
#             self.start_date = str(end_of_this_season - a_month)

#         self.end_date = str(today)

#     def test_fetch_betting_data(self):
#         data = fetch_betting_data(
#             start_date=self.start_date, end_date=self.end_date, verbose=0
#         )

#         self.assertIsInstance(data, list)
#         self.assertIsInstance(data[0], dict)
#         self.assertTrue(any(data))

#         dates = {datum["date"] for datum in data}
#         self.assertLessEqual(self.start_date, min(dates))
#         self.assertGreaterEqual(self.end_date, max(dates))


# class TestBettingDataProd(TestCase):
#     def setUp(self):
#         self.session = Session()
#         self.start_date = "2012-01-01"
#         self.end_date = "2013-12-31"

#     @patch.dict(os.environ, {**ENV_VARS, **{"PYTHON_ENV": "production"}}, clear=True)
#     @patch(f"{DATA_IMPORT_PATH}.betting_data.json.dump", MagicMock())
#     def test_fetch_betting_data(self):
#         with Betamax(self.session).use_cassette("betting_data"):
#             with patch(f"{DATA_IMPORT_PATH}.base_data.requests.get", self.session.get):
#                 data = fetch_betting_data(
#                     start_date=self.start_date, end_date=self.end_date, verbose=0
#                 )
#                 self.assertIsInstance(data, list)
#                 self.assertIsInstance(data[0], dict)
#                 self.assertTrue(any(data))

#                 dates = {datum["date"] for datum in data}
#                 self.assertLessEqual(self.start_date, min(dates))
#                 self.assertGreaterEqual(self.end_date, max(dates))


import atexit

from pact import Consumer, Provider, EachLike, Term, Like, matchers


pact = Consumer("Consumer").has_pact_with(Provider("Provider"))
pact.start_service()
atexit.register(pact.stop_service)


class TestBettingDataContract(TestCase):
    @patch.dict(
        os.environ,
        {
            **ENV_VARS,
            **{"PYTHON_ENV": "production", "AFL_DATA_SERVICE": "http://localhost:1234"},
        },
        clear=True,
    )
    def test_fetch_betting_data(self):
        start_date = "2018-01-01"
        end_date = "2018-12-31"

        expected = EachLike(
            {
                "date": Term(r"2018-\d{2}-\d{2}", "2018-05-21"),
                "venue": Like("MCG"),
                "round": Term(r"2018 Round \d{1,2}", "2018 Round 5"),
                "round_number": Like(5),
                "season": 2018,
                "home_team": Like("Blues"),
                "away_team": Like("Tigers"),
                "home_score": Like(64),
                "away_score": Like(97),
                "home_margin": Like(-33),
                "away_margin": Like(33),
                "home_win_odds": Like(6.75),
                "away_win_odds": Like(1.12),
                "home_win_paid": Like(0.0),
                "away_win_paid": Like(1.12),
                "home_line_odds": Like(39.5),
                "away_line_odds": Like(-39.5),
                "home_line_paid": Like(1.91),
                "away_line_paid": Like(0.0),
            }
        )

        (
            pact.given("Betting data exists")
            .upon_receiving("a request for betting data")
            .with_request(
                "get",
                "/betting_odds",
                query={"start_date": start_date, "end_date": end_date},
            )
            .will_respond_with(200, body=expected)
        )

        with pact:
            result = fetch_betting_data(start_date=start_date, end_date=end_date)

        self.assertEqual(result, matchers.get_generated_values(expected))
