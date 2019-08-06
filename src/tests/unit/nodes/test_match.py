from unittest import TestCase
import os

import pandas as pd

from machine_learning.nodes import match
from machine_learning.settings import BASE_DIR
from tests.fixtures.data_factories import fake_cleaned_match_data


TEST_DATA_DIR = os.path.join(BASE_DIR, "src/tests/fixtures")
MATCH_COUNT_PER_YEAR = 10
YEAR_RANGE = (2015, 2016)


class TestMatch(TestCase):
    def setUp(self):
        self.data_frame = fake_cleaned_match_data(MATCH_COUNT_PER_YEAR, YEAR_RANGE)

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

        self.__make_column_assertions(
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

    @staticmethod
    def __assert_column_added(
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
                with test_case.assertRaises(AssertionError):
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
