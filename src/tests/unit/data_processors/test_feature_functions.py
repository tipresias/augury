from unittest import TestCase

from faker import Faker
import numpy as np

from tests.fixtures.data_factories import fake_cleaned_match_data
from machine_learning.data_processors.feature_functions import (
    add_last_year_brownlow_votes,
    add_rolling_player_stats,
    add_cum_matches_played,
)


FAKE = Faker()
MATCH_COUNT_PER_YEAR = 10
YEAR_RANGE = (2015, 2016)
# Need to multiply by two, because we add team & oppo_team row per match
TOTAL_ROWS = MATCH_COUNT_PER_YEAR * len(range(*YEAR_RANGE)) * 2


def assert_column_added(
    test_case, column_names=[], valid_data_frame=None, feature_function=None, col_diff=1
):

    for column_name in column_names:
        with test_case.subTest(data_frame=valid_data_frame):
            data_frame = valid_data_frame
            transformed_data_frame = feature_function(data_frame)

            test_case.assertEqual(
                len(data_frame.columns) + col_diff, len(transformed_data_frame.columns)
            )
            test_case.assertIn(column_name, transformed_data_frame.columns)


def assert_required_columns(
    test_case, req_cols=[], valid_data_frame=None, feature_function=None
):
    for req_col in req_cols:
        with test_case.subTest(data_frame=valid_data_frame.drop(req_col, axis=1)):
            data_frame = valid_data_frame.drop(req_col, axis=1)
            with test_case.assertRaises(ValueError):
                feature_function(data_frame)


def make_column_assertions(
    test_case,
    column_names=[],
    req_cols=[],
    valid_data_frame=None,
    feature_function=None,
    col_diff=1,
):
    assert_column_added(
        test_case,
        column_names=column_names,
        valid_data_frame=valid_data_frame,
        feature_function=feature_function,
        col_diff=col_diff,
    )

    assert_required_columns(
        test_case,
        req_cols=req_cols,
        valid_data_frame=valid_data_frame,
        feature_function=feature_function,
    )


class TestFeatureFunctions(TestCase):
    def setUp(self):
        self.data_frame = fake_cleaned_match_data(MATCH_COUNT_PER_YEAR, YEAR_RANGE)

    def test_add_last_year_brownlow_votes(self):
        feature_function = add_last_year_brownlow_votes
        valid_data_frame = self.data_frame.assign(
            player_id=np.random.randint(100, 1000, TOTAL_ROWS),
            brownlow_votes=np.random.randint(0, 20, TOTAL_ROWS),
        )

        make_column_assertions(
            self,
            column_names=["last_year_brownlow_votes"],
            req_cols=("player_id", "year", "brownlow_votes"),
            valid_data_frame=valid_data_frame,
            feature_function=feature_function,
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

        feature_function = add_rolling_player_stats
        valid_data_frame = self.data_frame.assign(
            **{
                stats_col: np.random.randint(0, 20, TOTAL_ROWS)
                for stats_col in STATS_COLS
            }
        )

        make_column_assertions(
            self,
            column_names=[
                f"rolling_prev_match_{stats_col}"
                for stats_col in STATS_COLS
                if stats_col != "player_id"
            ],
            req_cols=STATS_COLS,
            valid_data_frame=valid_data_frame,
            feature_function=feature_function,
            col_diff=0,
        )

    def test_add_cum_matches_played(self):
        feature_function = add_cum_matches_played
        valid_data_frame = self.data_frame.assign(
            player_id=np.random.randint(100, 1000, TOTAL_ROWS)
        )

        make_column_assertions(
            self,
            column_names=["cum_matches_played"],
            req_cols=("player_id",),
            valid_data_frame=valid_data_frame,
            feature_function=feature_function,
        )
