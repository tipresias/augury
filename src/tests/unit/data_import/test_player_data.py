import os
from unittest import TestCase
from unittest.mock import patch, mock_open
import json

from tests.fixtures.data_factories import fake_cleaned_player_data
from augury.settings import RAW_DATA_DIR
from augury.data_import.player_data import save_player_data


START_DATE = "2012-01-01"
START_YEAR = int(START_DATE[:4])
END_DATE = "2013-12-31"
END_YEAR = int(END_DATE[:4]) + 1
N_MATCHES_PER_YEAR = 2
N_PLAYERS_PER_TEAM = 5
PLAYER_DATA_MODULE_PATH = "augury.data_import.player_data"
PLAYER_DATA_PATH = os.path.join(
    RAW_DATA_DIR, f"player-data_{START_DATE}_{END_DATE}.json"
)


class TestPlayerData(TestCase):
    def setUp(self):
        self.fake_player_data = fake_cleaned_player_data(
            N_MATCHES_PER_YEAR, (START_YEAR, END_YEAR), N_PLAYERS_PER_TEAM
        ).to_dict("records")

    @patch(f"{PLAYER_DATA_MODULE_PATH}.fetch_player_data")
    @patch("builtins.open", mock_open())
    @patch("json.dump")
    def test_save_player_data(self, _mock_json_dump, mock_fetch_data):
        mock_fetch_data.return_value = self.fake_player_data

        save_player_data(start_date=START_DATE, end_date=END_DATE, verbose=0)

        mock_fetch_data.assert_called_with(
            start_date=START_DATE, end_date=END_DATE, verbose=0
        )
        open.assert_called_with(PLAYER_DATA_PATH, "w")
        dump_args, _dump_kwargs = json.dump.call_args
        self.assertIn(self.fake_player_data, dump_args)
