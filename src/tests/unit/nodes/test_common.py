from unittest import TestCase
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np

from tests.fixtures.data_factories import (
    fake_raw_match_results_data,
    fake_footywire_betting_data,
    fake_cleaned_match_data,
)
from machine_learning.nodes import common
from machine_learning.settings import MELBOURNE_TIMEZONE, INDEX_COLS
from .node_test_mixins import ColumnAssertionMixin

START_DATE = "2013-01-01"
END_DATE = "2014-12-31"
START_YEAR = int(START_DATE[:4])
END_YEAR = int(END_DATE[:4]) + 1
N_MATCHES_PER_SEASON = 4
YEAR_RANGE = (START_YEAR, END_YEAR)
REQUIRED_OUTPUT_COLS = ["home_team", "year", "round_number"]

# Need to multiply by two, because we add team & oppo_team row per match
N_TEAMMATCH_ROWS = N_MATCHES_PER_SEASON * len(range(*YEAR_RANGE)) * 2


class TestCommon(TestCase, ColumnAssertionMixin):
    def setUp(self):
        self.data_frame = fake_cleaned_match_data(N_MATCHES_PER_SEASON, YEAR_RANGE)

    def test_convert_to_data_frame(self):
        data = fake_raw_match_results_data(
            N_MATCHES_PER_SEASON, (START_YEAR, END_YEAR)
        ).to_dict("records")

        data_frames = common.convert_to_data_frame(data, data)

        self.assertEqual(len(data_frames), 2)

        for data_frame in data_frames:
            self.assertIsInstance(data_frame, pd.DataFrame)

        raw_data_fields = data[0].keys()
        data_frame_columns = data_frames[0].columns

        self.assertEqual(set(raw_data_fields), set(data_frame_columns))

    def test_combine_data(self):
        raw_betting_data = fake_footywire_betting_data(
            N_MATCHES_PER_SEASON, YEAR_RANGE, clean=False
        )
        min_year_range = min(YEAR_RANGE)
        older_data = fake_footywire_betting_data(
            N_MATCHES_PER_SEASON, (min_year_range - 2, min_year_range), clean=False
        ).append(raw_betting_data.query("season == @min_year_range"))

        combine_data_func = common.combine_data(axis=0)
        combined_data = combine_data_func(raw_betting_data, older_data)

        total_year_range = range(min_year_range - 2, max(YEAR_RANGE))
        self.assertEqual({*total_year_range}, {*combined_data["season"]})

        self.assertEqual(
            N_MATCHES_PER_SEASON * len(total_year_range), len(combined_data)
        )

        with self.subTest(axis=1):
            match_year_range = (START_YEAR - 2, END_YEAR)
            match_data = fake_raw_match_results_data(
                N_MATCHES_PER_SEASON, match_year_range
            )

            combine_data_func = common.combine_data(axis=1)
            combined_data = combine_data_func(raw_betting_data, match_data)

            self.assertEqual(
                N_MATCHES_PER_SEASON * len(range(*match_year_range)), len(combined_data)
            )

            self.assertEqual(
                set(raw_betting_data.columns) | set(match_data.columns),
                set(combined_data.columns),
            )
            self.assertFalse((combined_data["date"] == 0).any())
            self.assertFalse(combined_data["date"].isna().any())

    def test_filter_by_date(self):
        raw_betting_data = fake_footywire_betting_data(
            N_MATCHES_PER_SEASON, YEAR_RANGE, clean=True
        )
        filter_start = f"{START_YEAR + 1}-06-01"
        filter_start_date = datetime.strptime(  # pylint: disable=unused-variable
            filter_start, "%Y-%m-%d"
        ).replace(tzinfo=MELBOURNE_TIMEZONE)
        filter_end = f"{START_YEAR + 1}-06-30"
        filter_end_date = datetime.strptime(  # pylint: disable=unused-variable
            filter_end, "%Y-%m-%d"
        ).replace(tzinfo=MELBOURNE_TIMEZONE)

        filter_func = common.filter_by_date(filter_start, filter_end)
        filtered_data_frame = filter_func(raw_betting_data)

        self.assertFalse(
            filtered_data_frame.query(
                "date < @filter_start_date | date > @filter_end_date"
            )
            .any()
            .any()
        )

        with self.subTest("with invalid date strings"):
            with self.assertRaises(ValueError):
                common.filter_by_date("what", "the what?")

        with self.subTest("without a date column"):
            with self.assertRaises(AssertionError):
                filter_func(raw_betting_data.drop("date", axis=1))

    def test_convert_match_rows_to_teammatch_rows(self):
        # DataFrame w/ minimum valid columns
        valid_data_frame = fake_cleaned_match_data(
            N_MATCHES_PER_SEASON, YEAR_RANGE, oppo_rows=False
        ).rename(
            columns={
                "team": "home_team",
                "oppo_team": "away_team",
                "score": "home_score",
                "oppo_score": "away_score",
            }
        )

        invalid_data_frame = valid_data_frame.drop("year", axis=1)

        with self.subTest(data_frame=valid_data_frame):
            transformed_df = common.convert_match_rows_to_teammatch_rows(
                valid_data_frame
            )

            self.assertIsInstance(transformed_df, pd.DataFrame)
            # TeamDataStacker stacks home & away teams, so the new DF should have twice as many rows
            self.assertEqual(len(valid_data_frame) * 2, len(transformed_df))
            # 'home_'/'away_' columns become regular columns or 'oppo_' columns,
            # non-team-specific columns are unchanged, and we add 'at_home'
            self.assertEqual(
                len(valid_data_frame.columns) + 1, len(transformed_df.columns)
            )
            self.assertIn("at_home", transformed_df.columns)
            # Half the teams should be marked as 'at_home'
            self.assertEqual(transformed_df["at_home"].sum(), len(transformed_df) / 2)

        with self.subTest(data_frame=invalid_data_frame):
            with self.assertRaises(AssertionError):
                common.convert_match_rows_to_teammatch_rows(invalid_data_frame)

    def test_add_oppo_features(self):
        REQUIRED_COLS = INDEX_COLS + ["oppo_team"]

        match_cols = [
            "date",
            "team",
            "oppo_team",
            "score",
            "oppo_score",
            "year",
            "round_number",
        ]
        oppo_feature_cols = ["kicks", "marks"]
        valid_data_frame = self.data_frame.assign(
            kicks=np.random.randint(50, 100, N_TEAMMATCH_ROWS),
            marks=np.random.randint(50, 100, N_TEAMMATCH_ROWS),
        )

        with self.subTest(data_frame=valid_data_frame, match_cols=match_cols):
            data_frame = valid_data_frame
            transform_func = common.add_oppo_features(match_cols=match_cols)
            transformed_df = transform_func(data_frame)

            # OppoFeatureBuilder adds 1 column per non-match column
            self.assertEqual(len(data_frame.columns) + 2, len(transformed_df.columns))

            # Should add the two new oppo columns
            self.assertIn("oppo_kicks", transformed_df.columns)
            self.assertIn("oppo_marks", transformed_df.columns)

            # Shouldn't add the match columns
            for match_col in match_cols:
                if match_col not in ["team", "score"]:
                    self.assertNotIn(f"oppo_{match_col}", transformed_df.columns)

            self.assertEqual(Counter(transformed_df.columns)["oppo_team"], 1)
            self.assertEqual(Counter(transformed_df.columns)["oppo_score"], 1)

            # Columns & their 'oppo_' equivalents should have the same values
            self.assertEqual(
                len(
                    np.setdiff1d(transformed_df["kicks"], transformed_df["oppo_kicks"])
                ),
                0,
            )
            self.assertEqual(
                len(
                    np.setdiff1d(transformed_df["marks"], transformed_df["oppo_marks"])
                ),
                0,
            )

        with self.subTest(
            data_frame=valid_data_frame, oppo_feature_cols=oppo_feature_cols
        ):
            data_frame = valid_data_frame
            transform_func = common.add_oppo_features(
                oppo_feature_cols=oppo_feature_cols
            )
            transformed_df = transform_func(data_frame)

            # OppoFeatureBuilder adds 1 column per non-match column
            self.assertEqual(len(data_frame.columns) + 2, len(transformed_df.columns))

            # Should add the two new oppo columns
            self.assertIn("oppo_kicks", transformed_df.columns)
            self.assertIn("oppo_marks", transformed_df.columns)

            # Shouldn't add the match columns
            for match_col in match_cols:
                if match_col not in ["team", "score"]:
                    self.assertNotIn(f"oppo_{match_col}", transformed_df.columns)

            self.assertEqual(Counter(transformed_df.columns)["oppo_team"], 1)
            self.assertEqual(Counter(transformed_df.columns)["oppo_score"], 1)

            # Columns & their 'oppo_' equivalents should have the same values
            self.assertEqual(
                len(
                    np.setdiff1d(transformed_df["kicks"], transformed_df["oppo_kicks"])
                ),
                0,
            )
            self.assertEqual(
                len(
                    np.setdiff1d(transformed_df["marks"], transformed_df["oppo_marks"])
                ),
                0,
            )

        with self.subTest(match_cols=match_cols, oppo_feature_cols=oppo_feature_cols):
            with self.assertRaises(ValueError):
                transform_func = common.add_oppo_features(
                    match_cols=match_cols, oppo_feature_cols=oppo_feature_cols
                )

        self._assert_required_columns(
            req_cols=REQUIRED_COLS,
            valid_data_frame=valid_data_frame,
            feature_function=transform_func,
        )

    def test_finalize_data(self):
        data_frame = (
            fake_cleaned_match_data(N_MATCHES_PER_SEASON, YEAR_RANGE)
            .assign(nans=None)
            .astype({"year": "str"})
        )

        finalized_data_frame = common.finalize_data(data_frame)

        self.assertEqual(finalized_data_frame["year"].dtype, int)
        self.assertFalse(finalized_data_frame["nans"].isna().any())

    def test_sort_columns(self):
        sort_data_frame_func = common.sort_data_frame_columns()
        sorted_data_frame = sort_data_frame_func(self.data_frame)

        non_numeric_cols = {"date", "team", "oppo_team"}
        first_cols = set(sorted_data_frame.columns[:3])

        self.assertEqual(non_numeric_cols, non_numeric_cols & first_cols)

        with self.subTest("with category_cols argument"):
            category_cols = ["team", "oppo_team"]

            sort_data_frame_func = common.sort_data_frame_columns(category_cols)
            sorted_data_frame = sort_data_frame_func(self.data_frame)

            first_cols = set(sorted_data_frame.columns[:2])

            self.assertEqual(set(category_cols), set(category_cols) & first_cols)
