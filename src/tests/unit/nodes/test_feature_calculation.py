# pylint: disable=missing-module-docstring, missing-function-docstring
# pylint: disable=missing-class-docstring

from unittest import TestCase
from faker import Faker
import pandas as pd
from candystore import CandyStore

from augury.pipelines.nodes import feature_calculation, common
from augury.pipelines.match import nodes as match


FAKE = Faker()
YEAR_RANGE = (2015, 2016)


def assert_required_columns(
    test_case, req_cols=[], valid_data_frame=None, feature_function=None
):
    for req_col in req_cols:
        with test_case.subTest(data_frame=valid_data_frame.drop(req_col, axis=1)):
            data_frame = valid_data_frame.drop(req_col, axis=1)
            with test_case.assertRaises(ValueError):
                feature_function(data_frame)


class TestFeatureCalculations(TestCase):
    def setUp(self):
        self.data_frame = (
            CandyStore(seasons=YEAR_RANGE)
            .match_results()
            .pipe(match.clean_match_data)
            .pipe(common.convert_match_rows_to_teammatch_rows)
        )

    def test_feature_calculator(self):
        def calc_func(col):
            return lambda df: df[col].rename(f"new_{col}")

        calculators = [
            (calc_func, ["team", "year"]),
            (calc_func, ["round_number", "score"]),
        ]
        calc_function = feature_calculation.feature_calculator(calculators)
        calculated_data_frame = calc_function(self.data_frame)

        self.assertIsInstance(calculated_data_frame, pd.DataFrame)
        self.assertFalse(any(calculated_data_frame.columns.duplicated()))

        with self.subTest("with calculate_rolling_rate"):
            calculators = [(feature_calculation.calculate_rolling_rate, [("score",)])]

            with self.subTest("with a multi-indexed data frame"):
                multi_index_df = self.data_frame.set_index(
                    ["team", "year", "round_number"], drop=False
                )

                # It runs without error
                calc_function = feature_calculation.feature_calculator(calculators)
                calc_function(multi_index_df)

    def test_rolling_rate_filled_by_expanding_rate(self):
        groups = self.data_frame[["team", "score", "oppo_score"]].groupby("team")
        window = 10
        rolling_values = feature_calculation.rolling_rate_filled_by_expanding_rate(
            groups, window
        )

        # It doesn't have any blank values
        self.assertFalse(rolling_values.isna().any().any())

    def test_calculate_rolling_rate(self):
        calc_function = feature_calculation.calculate_rolling_rate(("score",))

        assert_required_columns(
            self,
            req_cols=("score",),
            valid_data_frame=self.data_frame,
            feature_function=calc_function,
        )

        rolling_score = calc_function(self.data_frame)
        self.assertIsInstance(rolling_score, pd.Series)
        self.assertEqual(rolling_score.name, "rolling_score_rate")

    def test_calculate_division(self):
        calc_function = feature_calculation.calculate_division(("score", "oppo_score"))

        assert_required_columns(
            self,
            req_cols=("score", "oppo_score"),
            valid_data_frame=self.data_frame,
            feature_function=calc_function,
        )

        divided_scores = calc_function(self.data_frame)
        self.assertIsInstance(divided_scores, pd.Series)
        self.assertEqual(divided_scores.name, "score_divided_by_oppo_score")

    def test_calculate_multiplication(self):
        calc_function = feature_calculation.calculate_multiplication(
            ("score", "oppo_score")
        )

        assert_required_columns(
            self,
            req_cols=("score", "oppo_score"),
            valid_data_frame=self.data_frame,
            feature_function=calc_function,
        )

        multiplied_scores = calc_function(self.data_frame)
        self.assertIsInstance(multiplied_scores, pd.Series)
        self.assertEqual(multiplied_scores.name, "score_multiplied_by_oppo_score")

    def test_calculate_rolling_mean_by_dimension(self):
        calc_function = feature_calculation.calculate_rolling_mean_by_dimension(
            ("oppo_team", "score")
        )

        assert_required_columns(
            self,
            req_cols=("oppo_team", "score"),
            valid_data_frame=self.data_frame,
            feature_function=calc_function,
        )

        rolling_oppo_team_score = calc_function(self.data_frame)
        self.assertIsInstance(rolling_oppo_team_score, pd.Series)
        self.assertEqual(
            rolling_oppo_team_score.name, "rolling_mean_score_by_oppo_team"
        )

    def test_calculate_addition(self):
        calc_function = feature_calculation.calculate_addition(("score", "oppo_score"))

        assert_required_columns(
            self,
            req_cols=("score", "oppo_score"),
            valid_data_frame=self.data_frame,
            feature_function=calc_function,
        )

        addition_scores = calc_function(self.data_frame)
        self.assertIsInstance(addition_scores, pd.Series)
        self.assertEqual(addition_scores.name, "score_plus_oppo_score")
