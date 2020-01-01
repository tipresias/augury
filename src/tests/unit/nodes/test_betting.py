from unittest import TestCase
from datetime import time

import numpy as np
import pytz

from tests.fixtures.data_factories import (
    fake_footywire_betting_data,
    fake_cleaned_match_data,
)
from tests.helpers import ColumnAssertionMixin
from augury.nodes import betting


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
        clean_data = betting.clean_data(self.raw_betting_data)

        self.assertIn("year", clean_data.columns)

        invalid_cols = clean_data.filter(regex="_paid|_margin|venue|^round$").columns
        self.assertFalse(any(invalid_cols))
        self.assertEqual(
            {*REQUIRED_OUTPUT_COLS}, {*clean_data.columns} & {*REQUIRED_OUTPUT_COLS}
        )

        self.assertEqual(clean_data["date"].dt.tz, pytz.UTC)
        self.assertFalse((clean_data["date"].dt.time == time()).any())

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
            column_names=["betting_pred_win"],
            req_cols=("win_odds", "oppo_win_odds", "line_odds", "oppo_line_odds"),
            valid_data_frame=valid_data_frame,
            feature_function=feature_function,
        )
