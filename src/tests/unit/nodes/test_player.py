from unittest import TestCase
import os

import pandas as pd

from tests.fixtures.data_factories import fake_cleaned_player_data
from machine_learning.nodes import player
from machine_learning.settings import BASE_DIR


N_MATCHES_PER_SEASON = 10
YEAR_RANGE = (2015, 2016)
TEST_DATA_DIR = os.path.join(BASE_DIR, "src/tests/fixtures")
N_PLAYERS_PER_TEAM = 10


class TestPlayer(TestCase):
    def setUp(self):
        self.data_frame = fake_cleaned_player_data(
            N_MATCHES_PER_SEASON, YEAR_RANGE, N_PLAYERS_PER_TEAM
        )

    def test_clean_player_data(self):
        player_data = pd.read_csv(
            os.path.join(TEST_DATA_DIR, "fitzroy_get_afltables_stats.csv")
        )
        match_data = pd.read_csv(
            os.path.join(TEST_DATA_DIR, "fitzroy_match_results.csv")
        )

        clean_data = player.clean_player_data(player_data, match_data)

        self.assertIsInstance(clean_data, pd.DataFrame)

        required_columns = ["home_team", "away_team", "year", "round_number"]

        for col in required_columns:
            self.assertTrue(col in clean_data.columns.values)
