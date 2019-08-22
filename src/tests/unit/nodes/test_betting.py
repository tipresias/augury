from unittest import TestCase

import numpy as np

from tests.fixtures.data_factories import (
    fake_footywire_betting_data,
    fake_cleaned_match_data,
)
from machine_learning.nodes import betting
from .node_test_mixins import ColumnAssertionMixin


N_MATCHES_PER_SEASON = 4
START_YEAR = 2013
END_YEAR = 2015
YEAR_RANGE = (2013, 2015)
REQUIRED_OUTPUT_COLS = ["home_team", "year", "round_number"]

# Need to multiply by two, because we add team & oppo_team row per match
N_TEAMMATCH_ROWS = N_MATCHES_PER_SEASON * len(range(*YEAR_RANGE)) * 2


class TestBetting(TestCase, ColumnAssertionMixin):
    def setUp(self):
        self.raw_betting_data = fake_footywire_betting_data(
            N_MATCHES_PER_SEASON, YEAR_RANGE, clean=False
        )

    def test_clean_data(self):
        data = betting.clean_data(self.raw_betting_data)

        self.assertIn("year", data.columns)

        invalid_cols = data.filter(regex="_paid|_margin|venue|^round$").columns
        self.assertFalse(any(invalid_cols))
        self.assertEqual(
            {*REQUIRED_OUTPUT_COLS}, {*data.columns} & {*REQUIRED_OUTPUT_COLS}
        )

    def test_add_betting_pred_win(self):
        feature_function = betting.add_betting_pred_win

        valid_data_frame = fake_cleaned_match_data(
            N_MATCHES_PER_SEASON, YEAR_RANGE
        ).assign(
            win_odds=np.random.randint(0, 2, N_TEAMMATCH_ROWS),
            oppo_win_odds=np.random.randint(0, 2, N_TEAMMATCH_ROWS),
            line_odds=np.random.randint(-50, 50, N_TEAMMATCH_ROWS),
            oppo_line_odds=np.random.randint(-50, 50, N_TEAMMATCH_ROWS),
        )

        self._make_column_assertions(
            self,
            column_names=["betting_pred_win"],
            req_cols=("win_odds", "oppo_win_odds", "line_odds", "oppo_line_odds"),
            valid_data_frame=valid_data_frame,
            feature_function=feature_function,
        )
