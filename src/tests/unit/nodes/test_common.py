# pylint: disable=missing-module-docstring, missing-function-docstring
# pylint: disable=missing-class-docstring

from unittest import TestCase
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np
from candystore import CandyStore

from tests.helpers import ColumnAssertionMixin
from augury.nodes import common, base, match
from augury.settings import MELBOURNE_TIMEZONE, INDEX_COLS

START_DATE = "2013-01-01"
END_DATE = "2014-12-31"
START_YEAR = int(START_DATE[:4])
END_YEAR = int(END_DATE[:4]) + 1
YEAR_RANGE = (START_YEAR, END_YEAR)
REQUIRED_OUTPUT_COLS = ["home_team", "year", "round_number"]


class TestCommon(TestCase, ColumnAssertionMixin):
    def setUp(self):
        self.data_frame = (
            CandyStore(seasons=YEAR_RANGE)
            .match_results(to_dict=None)
            .pipe(match.clean_match_data)
            .pipe(common.convert_match_rows_to_teammatch_rows)
        )

    def test_convert_to_data_frame(self):
        data = CandyStore(seasons=(START_YEAR, END_YEAR)).match_results()
        data_frames = common.convert_to_data_frame(data, data)

        self.assertEqual(len(data_frames), 2)

        for data_frame in data_frames:
            self.assertIsInstance(data_frame, pd.DataFrame)

        raw_data_fields = data[0].keys()
        data_frame_columns = data_frames[0].columns

        self.assertEqual(set(raw_data_fields), set(data_frame_columns))

        with self.subTest("when data is empty"):
            data = []

            data_frames = common.convert_to_data_frame(data)

            # It is an empty data frame with no columns
            self.assertIsInstance(data_frames, pd.DataFrame)
            self.assertEqual(len(data_frames), 0)
            self.assertFalse(any(data_frames.columns))

    def test_combine_data(self):
        raw_betting_data = CandyStore(seasons=YEAR_RANGE).betting_odds(to_dict=None)
        min_year_range = min(YEAR_RANGE)
        older_data = (
            CandyStore(seasons=(min_year_range - 2, min_year_range))
            .betting_odds(to_dict=None)
            .append(raw_betting_data.query("season == @min_year_range"))
        )

        combine_data_func = common.combine_data(axis=0)
        combined_data = combine_data_func(raw_betting_data, older_data)

        total_year_range = range(min_year_range - 2, max(YEAR_RANGE))
        self.assertEqual({*total_year_range}, {*combined_data["season"]})

        expected_row_count = len(
            raw_betting_data.query("season != @min_year_range")
        ) + len(older_data)

        self.assertEqual(expected_row_count, len(combined_data))

        with self.subTest(axis=1):
            match_year_range = (START_YEAR - 2, END_YEAR)
            match_data = CandyStore(seasons=match_year_range).match_results(
                to_dict=None
            )

            combine_data_func = common.combine_data(axis=1)
            combined_data = combine_data_func(raw_betting_data, match_data)

            self.assertEqual(len(match_data), len(combined_data))

            self.assertEqual(
                set(raw_betting_data.columns) | set(match_data.columns),
                set(combined_data.columns),
            )
            self.assertFalse((combined_data["date"] == 0).any())
            self.assertFalse(combined_data["date"].isna().any())

    def test_filter_by_date(self):
        raw_betting_data = (
            CandyStore(seasons=YEAR_RANGE)
            .betting_odds(to_dict=None)
            .assign(date=base._parse_dates)  # pylint: disable=protected-access
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
        valid_data_frame = (
            CandyStore(seasons=YEAR_RANGE)
            .match_results(to_dict=None)
            .pipe(match.clean_match_data)
        )

        invalid_data_frame = valid_data_frame.drop("year", axis=1)

        with self.subTest(data_frame=valid_data_frame):
            transformed_df = common.convert_match_rows_to_teammatch_rows(
                valid_data_frame
            )

            self.assertIsInstance(transformed_df, pd.DataFrame)
            # TeamDataStacker stacks home & away teams, so the new DF should have
            # twice as many rows
            self.assertEqual(len(valid_data_frame) * 2, len(transformed_df))
            # 'home_'/'away_' columns become regular columns or 'oppo_' columns,
            # match_id is dropped, but otherwise non-team-specific columns
            # are unchanged, and we add 'at_home' (we drop & add a column,
            # so they should be equal)
            self.assertEqual(len(valid_data_frame.columns), len(transformed_df.columns))
            self.assertIn("at_home", transformed_df.columns)
            self.assertNotIn("match_id", transformed_df.columns)
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

        valid_data_frame = self.data_frame.loc[:, match_cols].assign(
            kicks=np.random.randint(50, 100, len(self.data_frame)),
            marks=np.random.randint(50, 100, len(self.data_frame)),
        )

        with self.subTest(data_frame=valid_data_frame, match_cols=match_cols):
            data_frame = valid_data_frame
            transform_func = common.add_oppo_features(match_cols=match_cols)
            transformed_df = transform_func(data_frame)

            # OppoFeatureBuilder adds 1 column per non-match column
            self.assertEqual(
                len(valid_data_frame.columns) + 2, len(transformed_df.columns)
            )

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
        data_frame = self.data_frame.assign(nans=None).astype({"year": "str"})

        finalized_data_frame = common.finalize_data(data_frame)

        self.assertEqual(finalized_data_frame["year"].dtype, int)
        self.assertFalse(finalized_data_frame["nans"].isna().any())

    def test_sort_columns(self):
        sort_data_frame_func = common.sort_data_frame_columns()
        sorted_data_frame = sort_data_frame_func(self.data_frame)

        non_numeric_cols = {"team", "oppo_team", "venue", "round_type", "date"}
        first_cols = set(sorted_data_frame.columns[slice(len(non_numeric_cols))])

        self.assertEqual(non_numeric_cols, non_numeric_cols & first_cols)

        with self.subTest("with category_cols argument"):
            category_cols = ["team", "oppo_team"]

            sort_data_frame_func = common.sort_data_frame_columns(category_cols)
            sorted_data_frame = sort_data_frame_func(self.data_frame)

            first_cols = set(sorted_data_frame.columns[:2])

            self.assertEqual(set(category_cols), set(category_cols) & first_cols)
