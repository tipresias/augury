# pylint: disable=missing-module-docstring, missing-function-docstring
# pylint: disable=missing-class-docstring

from unittest import TestCase
from datetime import time

import numpy as np
import pytz
from candystore import CandyStore

from tests.helpers import ColumnAssertionMixin
from augury.nodes import betting


YEAR_RANGE = (2013, 2015)
REQUIRED_OUTPUT_COLS = ["home_team", "year", "round_number"]


class TestBetting(TestCase, ColumnAssertionMixin):
    def setUp(self):
        self.raw_betting_data = CandyStore(seasons=YEAR_RANGE).betting_odds(
            to_dict=None
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
        match_data = CandyStore(seasons=YEAR_RANGE).match_results(to_dict=None)

        valid_data_frame = match_data.assign(
            win_odds=np.random.randint(0, 2, len(match_data)),
            oppo_win_odds=np.random.randint(0, 2, len(match_data)),
            line_odds=np.random.randint(-50, 50, len(match_data)),
            oppo_line_odds=np.random.randint(-50, 50, len(match_data)),
        )

        self._make_column_assertions(
            column_names=["betting_pred_win"],
            req_cols=("win_odds", "oppo_win_odds", "line_odds", "oppo_line_odds"),
            valid_data_frame=valid_data_frame,
            feature_function=feature_function,
        )
