# pylint: disable=missing-module-docstring, missing-function-docstring
# pylint: disable=missing-class-docstring

import os
from unittest import TestCase
from unittest.mock import patch, mock_open
import json

from candystore import CandyStore

from augury.settings import RAW_DATA_DIR
from augury.data_import.match_data import save_match_data


START_DATE = "2012-01-01"
START_YEAR = int(START_DATE[:4])
END_DATE = "2013-12-31"
END_YEAR = int(END_DATE[:4]) + 1
N_MATCHES_PER_YEAR = 2
MATCH_DATA_MODULE_PATH = "augury.data_import.match_data"
MATCH_DATA_PATH = os.path.join(RAW_DATA_DIR, f"match-data_{START_DATE}_{END_DATE}.json")


class TestMatchData(TestCase):
    def setUp(self):
        self.fake_match_data = CandyStore(
            seasons=(START_YEAR, END_YEAR)
        ).match_results()

    @patch(f"{MATCH_DATA_MODULE_PATH}.fetch_match_data")
    @patch("builtins.open", mock_open())
    @patch("json.dump")
    def test_save_match_data(self, _mock_json_dump, mock_fetch_data):
        mock_fetch_data.return_value = self.fake_match_data

        save_match_data(start_date=START_DATE, end_date=END_DATE, verbose=0)

        mock_fetch_data.assert_called_with(
            start_date=START_DATE, end_date=END_DATE, verbose=0
        )
        open.assert_called_with(MATCH_DATA_PATH, "w", encoding="utf-8")
        dump_args, _dump_kwargs = json.dump.call_args
        self.assertIn(self.fake_match_data, dump_args)
