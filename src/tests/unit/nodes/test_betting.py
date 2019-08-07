from unittest import TestCase

import numpy as np

from tests.fixtures.data_factories import (
    fake_footywire_betting_data,
    fake_cleaned_match_data,
)
from machine_learning.nodes import betting


N_MATCHES_PER_SEASON = 4
START_YEAR = 2013
END_YEAR = 2015
YEAR_RANGE = (2013, 2015)
REQUIRED_OUTPUT_COLS = ["home_team", "year", "round_number"]

# Need to multiply by two, because we add team & oppo_team row per match
N_TEAMMATCH_ROWS = N_MATCHES_PER_SEASON * len(range(*YEAR_RANGE)) * 2


class TestBetting(TestCase):
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

        self.__make_column_assertions(
            self,
            column_names=["betting_pred_win"],
            req_cols=("win_odds", "oppo_win_odds", "line_odds", "oppo_line_odds"),
            valid_data_frame=valid_data_frame,
            feature_function=feature_function,
        )

    def test_finalize_data(self):
        data_frame = (
            fake_cleaned_match_data(N_MATCHES_PER_SEASON, YEAR_RANGE)
            .assign(nans=None)
            .astype({"year": "str"})
        )

        finalized_data_frame = betting.finalize_data(data_frame)

        self.assertEqual(finalized_data_frame["year"].dtype, int)
        self.assertFalse(finalized_data_frame["nans"].isna().any())

    def __assert_column_added(
        self,
        test_case,
        column_names=[],
        valid_data_frame=None,
        feature_function=None,
        col_diff=1,
    ):

        for column_name in column_names:
            with test_case.subTest(data_frame=valid_data_frame):
                data_frame = valid_data_frame
                transformed_data_frame = feature_function(data_frame)

                test_case.assertEqual(
                    len(data_frame.columns) + col_diff,
                    len(transformed_data_frame.columns),
                )
                test_case.assertIn(column_name, transformed_data_frame.columns)

    @staticmethod
    def __assert_required_columns(
        test_case, req_cols=[], valid_data_frame=None, feature_function=None
    ):
        for req_col in req_cols:
            with test_case.subTest(data_frame=valid_data_frame.drop(req_col, axis=1)):
                data_frame = valid_data_frame.drop(req_col, axis=1)
                with test_case.assertRaises(ValueError):
                    feature_function(data_frame)

    def __make_column_assertions(
        self,
        test_case,
        column_names=[],
        req_cols=[],
        valid_data_frame=None,
        feature_function=None,
        col_diff=1,
    ):
        self.__assert_column_added(
            test_case,
            column_names=column_names,
            valid_data_frame=valid_data_frame,
            feature_function=feature_function,
            col_diff=col_diff,
        )

        self.__assert_required_columns(
            test_case,
            req_cols=req_cols,
            valid_data_frame=valid_data_frame,
            feature_function=feature_function,
        )
