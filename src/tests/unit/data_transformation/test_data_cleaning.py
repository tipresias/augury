import os
from unittest import TestCase
import pandas as pd
from faker import Faker

from machine_learning.data_transformation.data_cleaning import clean_joined_data
from machine_learning.settings import BASE_DIR


TEST_DATA_DIR = os.path.join(BASE_DIR, "src/tests/fixtures")
N_PLAYERS = 20
N_MATCHES = 5
FAKE = Faker()


class TestDataCleaning(TestCase):
    def test_clean_joined_data(self):
        match_data = pd.read_csv(
            os.path.join(TEST_DATA_DIR, "fitzroy_match_results.csv")
        )
        player_data = pd.read_csv(
            os.path.join(TEST_DATA_DIR, "fitzroy_get_afltables_stats.csv")
        )
        betting_data = pd.read_json(os.path.join(TEST_DATA_DIR, "afl_betting.json"))

        clean_data = clean_joined_data([player_data, match_data, betting_data])

        self.assertIsInstance(clean_data, pd.DataFrame)

        are_duplicate_columns = pd.concat(
            [
                pd.Series(match_data.columns),
                pd.Series(player_data.columns),
                pd.Series(betting_data.columns),
            ]
        ).duplicated()
        self.assertTrue(are_duplicate_columns.any())
        are_duplicate_clean_columns = clean_data.columns.duplicated()
        self.assertFalse(are_duplicate_clean_columns.any())
