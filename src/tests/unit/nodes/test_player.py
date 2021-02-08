# pylint: disable=missing-module-docstring, missing-function-docstring
# pylint: disable=missing-class-docstring

from unittest import TestCase
import os
from datetime import time

import pandas as pd
from faker import Faker
import numpy as np
import pytz
from candystore import CandyStore

from tests.helpers import ColumnAssertionMixin
from augury.pipelines.player import nodes as player
from augury.pipelines.nodes import common
from augury.settings import INDEX_COLS, BASE_DIR


YEAR_RANGE = (2015, 2016)
TEST_DATA_DIR = os.path.join(BASE_DIR, "src/tests/fixtures")

FAKE = Faker()


class TestPlayer(TestCase, ColumnAssertionMixin):
    def setUp(self):
        self.data_frame = CandyStore(seasons=YEAR_RANGE).players()

    def test_clean_player_data(self):
        player_data = pd.read_csv(
            os.path.join(TEST_DATA_DIR, "fitzroy_get_afltables_stats.csv")
        )
        match_data = pd.read_csv(
            os.path.join(TEST_DATA_DIR, "fitzroy_match_results.csv")
        ).assign(match_id=lambda df: df.index.values)

        clean_data = player.clean_player_data(player_data, match_data)

        self.assertIsInstance(clean_data, pd.DataFrame)

        required_columns = ["home_team", "away_team", "year", "round_number"]

        for col in required_columns:
            self.assertTrue(col in clean_data.columns.values)

        self.assertEqual(clean_data["date"].dt.tz, pytz.UTC)
        self.assertFalse((clean_data["date"].dt.time == time()).any())

    def test_clean_roster_data(self):
        roster_data = pd.read_json(
            os.path.join(TEST_DATA_DIR, "team_rosters.json"), convert_dates=False
        )
        dummy_player_data = (
            CandyStore(seasons=YEAR_RANGE)
            .players()
            .assign(player_name=lambda df: df["first_name"] + " " + df["surname"])
            .drop(["first_name", "surname"], axis=1)
            .rename(columns={"id": "player_id"})
        )

        clean_data = player.clean_roster_data(roster_data, dummy_player_data)

        self.assertIsInstance(clean_data, pd.DataFrame)

        required_columns = ["home_team", "away_team", "year"]

        for col in required_columns:
            self.assertTrue(col in clean_data.columns.values)

        self.assertEqual(clean_data["date"].dt.tz, pytz.UTC)
        self.assertFalse((clean_data["date"].dt.time == time()).any())

    def test_add_last_year_brownlow_votes(self):
        valid_data_frame = self.data_frame.rename(
            columns={"season": "year", "id": "player_id"}
        )

        self._make_column_assertions(
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
                stats_col: np.random.randint(0, 20, len(self.data_frame))
                for stats_col in STATS_COLS
            }
        ).rename(columns={"season": "year", "round": "round_number"})

        self._make_column_assertions(
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
            player_id=np.random.randint(100, 1000, len(self.data_frame))
        )

        self._make_column_assertions(
            column_names=["cum_matches_played"],
            req_cols=("player_id",),
            valid_data_frame=valid_data_frame,
            feature_function=player.add_cum_matches_played,
        )

    def test_aggregate_player_stats_by_team_match(self):
        stats_col_assignments = {
            stats_col: np.random.randint(0, 20, len(self.data_frame))
            for stats_col in player.PLAYER_STATS_COLS
        }
        # Drop 'playing_for', because it gets dropped by PlayerDataStacker,
        # which comes before PlayerDataAggregator in the pipeline
        valid_data_frame = (
            self.data_frame.loc[
                :,
                [
                    "first_name",
                    "surname",
                    "round",
                    "season",
                    "home_team",
                    "away_team",
                    "id",
                    "date",
                    "home_score",
                    "away_score",
                ],
            ]
            .assign(
                **{
                    **{
                        "player_name": lambda df: df["first_name"] + " " + df["surname"]
                    },
                    **stats_col_assignments,
                }
            )
            .drop(["first_name", "surname"], axis=1)
            .rename(
                columns={
                    "round": "round_number",
                    "season": "year",
                    "id": "player_id",
                }
            )
            .astype({"player_id": str})
            .pipe(common.convert_match_rows_to_teammatch_rows)
        )

        aggregation_func = player.aggregate_player_stats_by_team_match(["sum", "mean"])

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
            req_cols=(
                INDEX_COLS
                + player.PLAYER_STATS_COLS
                + ["oppo_team", "player_id", "player_name", "date"]
            ),
            valid_data_frame=valid_data_frame,
            feature_function=aggregation_func,
        )
