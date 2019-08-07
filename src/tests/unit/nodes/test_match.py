from unittest import TestCase
import os

import pandas as pd
import numpy as np
from faker import Faker

from tests.fixtures.data_factories import fake_cleaned_match_data
from machine_learning.nodes import match
from machine_learning.settings import BASE_DIR
from machine_learning.data_config import VENUES
from .node_test_mixins import ColumnAssertionMixin


FAKE = Faker()

TEST_DATA_DIR = os.path.join(BASE_DIR, "src/tests/fixtures")
N_MATCHES_PER_SEASON = 10
YEAR_RANGE = (2015, 2016)

# Need to multiply by two, because we add team & oppo_team row per match
N_TEAMMATCH_ROWS = N_MATCHES_PER_SEASON * len(range(*YEAR_RANGE)) * 2


class TestMatch(TestCase, ColumnAssertionMixin):
    def setUp(self):
        self.data_frame = fake_cleaned_match_data(N_MATCHES_PER_SEASON, YEAR_RANGE)

    def test_clean_match_data(self):
        match_data = pd.read_csv(
            os.path.join(TEST_DATA_DIR, "fitzroy_match_results.csv")
        )

        clean_data = match.clean_match_data(match_data)

        self.assertIsInstance(clean_data, pd.DataFrame)

        required_columns = ["home_team", "away_team", "year", "round_number"]

        for col in required_columns:
            self.assertTrue(col in clean_data.columns.values)

    def test_clean_fixture_data(self):
        fixture_data = pd.read_csv(
            os.path.join(TEST_DATA_DIR, "ft_match_list.csv")
        ).query("season == 2019")

        clean_data = match.clean_fixture_data(fixture_data)

        self.assertIsInstance(clean_data, pd.DataFrame)
        self.assertFalse(clean_data.isna().any().any())

        required_columns = ["home_team", "away_team", "year", "round_number"]

        for col in required_columns:
            self.assertTrue(col in clean_data.columns.values)

    def test_add_elo_rating(self):
        valid_data_frame = self.data_frame.rename(
            columns={
                "team": "home_team",
                "oppo_team": "away_team",
                "score": "home_score",
                "oppo_score": "away_score",
            }
        )

        self._make_column_assertions(
            self,
            column_names=["home_elo_rating", "away_elo_rating"],
            req_cols=(
                "home_score",
                "away_score",
                "home_team",
                "away_team",
                "year",
                "date",
                "round_number",
            ),
            valid_data_frame=valid_data_frame,
            feature_function=match.add_elo_rating,
            col_diff=2,
        )

    def test_add_out_of_state(self):
        feature_function = match.add_out_of_state
        valid_data_frame = self.data_frame.assign(
            venue=[VENUES[idx % len(VENUES)] for idx in range(N_TEAMMATCH_ROWS)]
        )

        self._make_column_assertions(
            self,
            column_names=["out_of_state"],
            req_cols=("venue", "team"),
            valid_data_frame=valid_data_frame,
            feature_function=feature_function,
        )

    def test_add_travel_distance(self):
        feature_function = match.add_travel_distance
        valid_data_frame = self.data_frame.assign(
            venue=[VENUES[idx % len(VENUES)] for idx in range(N_TEAMMATCH_ROWS)]
        )

        self._make_column_assertions(
            self,
            column_names=["travel_distance"],
            req_cols=("venue", "team"),
            valid_data_frame=valid_data_frame,
            feature_function=feature_function,
        )

    def test_add_result(self):
        feature_function = match.add_result
        valid_data_frame = self.data_frame

        self._make_column_assertions(
            self,
            column_names=["result"],
            req_cols=("score", "oppo_score"),
            valid_data_frame=valid_data_frame,
            feature_function=feature_function,
        )

    def test_add_margin(self):
        feature_function = match.add_margin
        valid_data_frame = self.data_frame

        self._make_column_assertions(
            self,
            column_names=["margin"],
            req_cols=("score", "oppo_score"),
            valid_data_frame=valid_data_frame,
            feature_function=feature_function,
        )

    def test_add_shifted_team_features(self):
        feature_function = match.add_shifted_team_features(shift_columns=["score"])
        valid_data_frame = self.data_frame.assign(team=FAKE.company())

        self._make_column_assertions(
            self,
            column_names=["prev_match_score"],
            req_cols=("score",),
            valid_data_frame=valid_data_frame,
            feature_function=feature_function,
        )

        shifted_data_frame = feature_function(valid_data_frame)
        self.assertEqual(shifted_data_frame["prev_match_score"].iloc[0], 0)
        self.assertEqual(
            shifted_data_frame["prev_match_score"].iloc[1],
            shifted_data_frame["score"].iloc[0],
        )

        with self.subTest("using keep_columns argument"):
            keep_columns = [col for col in self.data_frame if col != "score"]
            feature_function = match.add_shifted_team_features(
                keep_columns=keep_columns
            )
            valid_data_frame = self.data_frame.assign(team=FAKE.company())

            self._assert_column_added(
                self,
                column_names=["prev_match_score"],
                valid_data_frame=valid_data_frame,
                feature_function=feature_function,
            )

            shifted_data_frame = feature_function(valid_data_frame)
            self.assertEqual(shifted_data_frame["prev_match_score"].iloc[0], 0)
            self.assertEqual(
                shifted_data_frame["prev_match_score"].iloc[1],
                shifted_data_frame["score"].iloc[0],
            )
            prev_match_columns = [
                col for col in shifted_data_frame.columns if "prev_match" in col
            ]
            self.assertEqual(len(prev_match_columns), 1)

    def test_add_cum_win_points(self):
        feature_function = match.add_cum_win_points
        valid_data_frame = self.data_frame.assign(
            prev_match_result=np.random.randint(0, 2, N_TEAMMATCH_ROWS)
        )

        self._make_column_assertions(
            self,
            column_names=["cum_win_points"],
            req_cols=("prev_match_result",),
            valid_data_frame=valid_data_frame,
            feature_function=feature_function,
        )

    def test_add_win_streak(self):
        feature_function = match.add_win_streak
        valid_data_frame = self.data_frame.assign(
            prev_match_result=np.random.randint(0, 2, N_TEAMMATCH_ROWS)
        )

        self._make_column_assertions(
            self,
            column_names=["win_streak"],
            req_cols=("prev_match_result",),
            valid_data_frame=valid_data_frame,
            feature_function=feature_function,
        )
