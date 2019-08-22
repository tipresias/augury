from unittest import TestCase
import os

import pandas as pd
from faker import Faker
import numpy as np

from tests.fixtures.data_factories import fake_cleaned_player_data
from machine_learning.nodes import player
from machine_learning.settings import BASE_DIR
from machine_learning.data_config import INDEX_COLS
from .node_test_mixins import ColumnAssertionMixin


N_MATCHES_PER_SEASON = 10
YEAR_RANGE = (2015, 2016)
TEST_DATA_DIR = os.path.join(BASE_DIR, "src/tests/fixtures")
N_PLAYERS_PER_TEAM = 10
# Need to multiply by two, because we add team & oppo_team row per match
N_TEAMMATCH_ROWS = N_MATCHES_PER_SEASON * len(range(*YEAR_RANGE)) * 2
N_PLAYER_ROWS = N_TEAMMATCH_ROWS * N_PLAYERS_PER_TEAM

FAKE = Faker()


class TestPlayer(TestCase, ColumnAssertionMixin):
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

    def test_clean_roster_data(self):
        roster_data = pd.read_json(os.path.join(TEST_DATA_DIR, "team_rosters.json"))
        dummy_player_data = pd.DataFrame(
            {
                "player_id": [FAKE.ean() for _ in range(len(roster_data))],
                "player_name": [FAKE.name() for _ in range(len(roster_data))],
            }
        )

        clean_data = player.clean_roster_data(roster_data, dummy_player_data)

        self.assertIsInstance(clean_data, pd.DataFrame)

        required_columns = ["home_team", "away_team", "year"]

        for col in required_columns:
            self.assertTrue(col in clean_data.columns.values)

    def test_add_last_year_brownlow_votes(self):
        valid_data_frame = self.data_frame.assign(
            player_id=np.random.randint(100, 1000, N_PLAYER_ROWS),
            brownlow_votes=np.random.randint(0, 20, N_PLAYER_ROWS),
        )

        self._make_column_assertions(
            self,
            column_names=["last_year_brownlow_votes"],
            req_cols=("player_id", "year", "brownlow_votes"),
            valid_data_frame=valid_data_frame,
            feature_function=player.add_last_year_brownlow_votes,
            col_diff=0,
        )

    def test_add_rolling_player_stats(self):
        STATS_COLS = [
            "player_id",
            "kicks",
            "marks",
            "handballs",
            "goals",
            "behinds",
            "hit_outs",
            "tackles",
            "rebounds",
            "inside_50s",
            "clearances",
            "clangers",
            "frees_for",
            "frees_against",
            "contested_possessions",
            "uncontested_possessions",
            "contested_marks",
            "marks_inside_50",
            "one_percenters",
            "bounces",
            "goal_assists",
            "time_on_ground",
        ]

        valid_data_frame = self.data_frame.assign(
            **{
                stats_col: np.random.randint(0, 20, N_PLAYER_ROWS)
                for stats_col in STATS_COLS
            }
        )

        self._make_column_assertions(
            self,
            column_names=[
                f"rolling_prev_match_{stats_col}"
                for stats_col in STATS_COLS
                if stats_col != "player_id"
            ],
            req_cols=STATS_COLS,
            valid_data_frame=valid_data_frame,
            feature_function=player.add_rolling_player_stats,
            col_diff=0,
        )

    def test_add_cum_matches_played(self):
        valid_data_frame = self.data_frame.assign(
            player_id=np.random.randint(100, 1000, N_PLAYER_ROWS)
        )

        self._make_column_assertions(
            self,
            column_names=["cum_matches_played"],
            req_cols=("player_id",),
            valid_data_frame=valid_data_frame,
            feature_function=player.add_cum_matches_played,
        )

    def test_aggregate_player_stats_by_team_match(self):
        stats_col_assignments = {
            stats_col: np.random.randint(0, 20, N_PLAYER_ROWS)
            for stats_col in player.PLAYER_STATS_COLS
        }
        # Drop 'playing_for', because it gets dropped by PlayerDataStacker,
        # which comes before PlayerDataAggregator in the pipeline
        valid_data_frame = self.data_frame.assign(**stats_col_assignments).drop(
            "playing_for", axis=1
        )

        aggregation_func = player.aggregate_player_stats_by_team_match(["sum", "std"])

        transformed_df = aggregation_func(valid_data_frame)

        self.assertIsInstance(transformed_df, pd.DataFrame)

        # We drop player_id & player_name, but add new stats cols for each aggregation
        expected_col_count = (
            len(valid_data_frame.columns) - 2 + len(player.PLAYER_STATS_COLS)
        )
        self.assertEqual(expected_col_count, len(transformed_df.columns))

        # Match data should remain unchanged (requires a little extra manipulation,
        # because I can't be bothered to make the score data realistic)
        for idx, value in enumerate(
            valid_data_frame.groupby(["team", "year", "round_number"])["score"]
            .mean()
            .astype(int)
        ):
            self.assertEqual(value, transformed_df["score"].iloc[idx])

        for idx, value in enumerate(
            valid_data_frame.groupby(["team", "year", "round_number"])["oppo_score"]
            .mean()
            .astype(int)
        ):
            self.assertEqual(value, transformed_df["oppo_score"].iloc[idx])

        # Player data should be aggregated, but same sum
        self.assertEqual(
            valid_data_frame["rolling_prev_match_kicks"].sum(),
            transformed_df["rolling_prev_match_kicks_sum"].sum(),
        )

        self._assert_required_columns(
            self,
            req_cols=(
                INDEX_COLS
                + player.PLAYER_STATS_COLS
                + ["oppo_team", "player_id", "player_name", "date"]
            ),
            valid_data_frame=valid_data_frame,
            feature_function=aggregation_func,
        )
