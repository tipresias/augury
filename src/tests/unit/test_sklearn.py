from unittest import TestCase
import os

from sklearn.linear_model import Ridge, Lasso
import pandas as pd
import numpy as np
from faker import Faker

from tests.helpers import KedroContextMixin
from tests.fixtures.data_factories import fake_cleaned_match_data
from tests.fixtures.fake_estimator import FakeEstimatorData
from augury.sklearn import (
    AveragingRegressor,
    CorrelationSelector,
    EloRegressor,
    TeammatchToMatchConverter,
    ColumnDropper,
    DataFrameConverter,
    MATCH_INDEX_COLS,
    match_accuracy_scorer,
    year_cv_split,
)
from augury.settings import BASE_DIR


FAKE = Faker()
ROW_COUNT = 10


class TestAveragingRegressor(TestCase):
    def setUp(self):
        data_frame = pd.DataFrame(
            {
                "year": ([2014] * round(ROW_COUNT * 0.2))
                + ([2015] * round(ROW_COUNT * 0.6))
                + ([2016] * round(ROW_COUNT * 0.2)),
                "prev_match_score": np.random.randint(50, 150, ROW_COUNT),
                "prev_match_oppo_score": np.random.randint(50, 150, ROW_COUNT),
                "round_number": 15,
                "margin": np.random.randint(5, 50, ROW_COUNT),
            }
        )

        self.X = data_frame.drop("margin", axis=1)
        self.y = data_frame["margin"]
        self.regressor = AveragingRegressor([("ridge", Ridge()), ("lasso", Lasso())])

    def test_predict(self):
        self.regressor.fit(self.X, self.y)
        predictions = self.regressor.predict(self.X)

        self.assertIsInstance(predictions, np.ndarray)


class TestCorrelationSelector(TestCase):
    def setUp(self):
        self.data_frame = pd.DataFrame(
            {
                "year": ([2014] * round(ROW_COUNT * 0.2))
                + ([2015] * round(ROW_COUNT * 0.6))
                + ([2016] * round(ROW_COUNT * 0.2)),
                "prev_match_score": np.random.randint(50, 150, ROW_COUNT),
                "prev_match_oppo_score": np.random.randint(50, 150, ROW_COUNT),
                "round_number": 15,
                "margin": np.random.randint(5, 50, ROW_COUNT),
            }
        )

        self.X = self.data_frame.drop("margin", axis=1)
        self.y = self.data_frame["margin"]
        self.selector = CorrelationSelector()

    def test_transform(self):
        transformed_data_frame = self.selector.fit_transform(self.X, self.y)

        self.assertIsInstance(transformed_data_frame, pd.DataFrame)
        self.assertEqual(len(transformed_data_frame.columns), len(self.X.columns))

        with self.subTest("threshold > 0"):
            label_correlations = (
                self.data_frame.corr().fillna(0)["margin"].abs().sort_values()
            )
            threshold = label_correlations.iloc[round(len(label_correlations) * 0.5)]

            self.selector.threshold = threshold
            transformed_data_frame = self.selector.fit_transform(self.X, self.y)

            self.assertLess(len(transformed_data_frame.columns), len(self.X.columns))

        with self.subTest("cols_to_keep not empty"):
            cols_to_keep = [
                col for col in self.X.columns if col not in transformed_data_frame
            ][:2]

            self.selector.cols_to_keep = cols_to_keep
            transformed_data_frame = self.selector.fit_transform(self.X, self.y)

            for col in cols_to_keep:
                self.assertIn(col, transformed_data_frame.columns)

        with self.subTest("empty labels argument"):
            self.selector = CorrelationSelector()

            with self.assertRaisesRegex(AssertionError, r"Need labels argument"):
                self.selector.fit_transform(self.X, pd.Series())


class TestEloRegressor(TestCase):
    def setUp(self):
        self.data_frame = (
            pd.read_json(os.path.join(BASE_DIR, "src/tests/fixtures/elo_data.json"))
            .set_index(["home_team", "year", "round_number"], drop=False)
            .rename_axis([None, None, None])
            .sort_index()
        )

        self.X = self.data_frame
        self.regressor = EloRegressor()

    def test_predict(self):
        X_train = self.X.query("year == 2014")
        X_test = self.X.query("year == 2015")
        # We don't use y when fitting the Elo model, so it can be just filler
        y = np.zeros(len(X_train))

        self.regressor.fit(X_train, y)
        predictions = self.regressor.predict(X_test)

        self.assertIsInstance(predictions, np.ndarray)

        with self.subTest("when there's a gap between match rounds"):
            invalid_X_test = X_test.query("round_number > 5")
            # Need to refit, because predict updates the state of the Elo ratings
            self.regressor.fit(X_train, y)

            with self.assertRaises(AssertionError):
                self.regressor.predict(invalid_X_test)


class TestTeammatchToMatchConverter(TestCase):
    def setUp(self):
        self.data = fake_cleaned_match_data(ROW_COUNT, (2017, 2018))
        self.data.loc[:, "at_home"] = [row % 2 for row in range(len(self.data))]
        self.match_cols = ["date", "year", "round_number"]
        self.transformer = TeammatchToMatchConverter(match_cols=self.match_cols)

    def test_transform(self):
        self.transformer.fit(self.data, None)
        transformed_data = self.transformer.transform(self.data)

        self.assertEqual(len(self.data), len(transformed_data) * 2)
        self.assertIn("home_team", transformed_data.columns)
        self.assertIn("away_team", transformed_data.columns)
        self.assertNotIn("oppo_team", transformed_data.columns)

        with self.subTest("when a match_col is missing"):
            invalid_data = self.data.drop("date", axis=1)

            with self.assertRaisesRegex(AssertionError, r"required columns"):
                self.transformer.transform(invalid_data)

        with self.subTest("when a match-index column is missing"):
            for match_col in MATCH_INDEX_COLS:
                invalid_data = self.data.drop(match_col, axis=1)

                with self.assertRaisesRegex(AssertionError, r"required columns"):
                    self.transformer.transform(invalid_data)


class TestColumnDropper(TestCase):
    def setUp(self):
        self.data = fake_cleaned_match_data(ROW_COUNT, (2017, 2018))
        self.cols_to_drop = ["team", "oppo_score"]
        self.transformer = ColumnDropper(cols_to_drop=self.cols_to_drop)

    def test_transform(self):
        self.transformer.fit(self.data, None)
        transformed_data = self.transformer.transform(self.data)

        for column in self.cols_to_drop:
            self.assertNotIn(column, transformed_data.columns)


class TestDataFrameConverter(TestCase):
    def setUp(self):
        self.data = fake_cleaned_match_data(ROW_COUNT, (2017, 2018))
        self.transformer = DataFrameConverter(
            columns=self.data.columns, index=self.data.index
        )

    def test_fit(self):
        self.assertEqual(self.transformer, self.transformer.fit(self.data))

    def test_transform(self):
        transformed_data = self.transformer.transform(self.data.to_numpy())
        self.assertIsInstance(transformed_data, pd.DataFrame)

        column_set = set(transformed_data.columns) & set(self.data.columns)
        self.assertEqual(column_set, set(self.data.columns))
        index_set = set(transformed_data.index) & set(self.data.index)
        self.assertEqual(index_set, set(self.data.index))

        with self.subTest("when index length doesn't match data shape"):
            self.transformer.set_params(index=self.data.iloc[2:, :].index)

            with self.assertRaisesRegex(
                AssertionError, "X must have the same number of rows"
            ):
                self.transformer.transform(self.data.to_numpy())

        with self.subTest("when column length doesn't match data shape"):
            self.transformer.set_params(
                index=self.data.index, columns=self.data.iloc[:, 2:].columns
            )

            with self.assertRaisesRegex(
                AssertionError, "X must have the same number of columns"
            ):
                self.transformer.transform(self.data.to_numpy())


class TestSklearn(TestCase, KedroContextMixin):
    def setUp(self):
        self.data = FakeEstimatorData()

    def test_match_accuracy_scorer(self):
        estimator = self.load_context().catalog.load("fake_estimator")
        X_test, y_test = self.data.test_data

        match_acc = match_accuracy_scorer(estimator, X_test, y_test)

        self.assertIsInstance(match_acc, float)
        self.assertGreater(match_acc, 0)

    def test_year_cv_split(self):
        max_train_year = max(self.data.train_year_range)
        n_splits = 5
        year_range = (max_train_year - n_splits, max_train_year)
        X_train, _ = self.data.train_data

        cv_splits = year_cv_split(X_train, year_range)

        self.assertIsInstance(cv_splits, list)
        self.assertEqual(len(cv_splits), n_splits)

        for split in cv_splits:
            self.assertIsInstance(split, tuple)
            self.assertEqual(len(split), 2)

            train, test = split
            self.assertFalse(train[test].any())
