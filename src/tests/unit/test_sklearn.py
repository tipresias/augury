# pylint: disable=missing-docstring

from unittest import TestCase
from unittest.mock import patch
import os
import warnings

from sklearn.linear_model import Ridge, Lasso, LogisticRegression
import pandas as pd
import numpy as np
from faker import Faker
from tensorflow import keras
import pytest
from candystore import CandyStore

from tests.helpers import KedroContextMixin
from tests.fixtures.fake_estimator import FakeEstimatorData, create_fake_pipeline
from augury.nodes import match, common
from augury.sklearn.models import AveragingRegressor, EloRegressor, KerasClassifier
from augury.sklearn.preprocessing import (
    CorrelationSelector,
    TeammatchToMatchConverter,
    ColumnDropper,
    DataFrameConverter,
    MATCH_INDEX_COLS,
)
from augury.sklearn.metrics import match_accuracy_scorer, bits_scorer, bits_objective
from augury.sklearn.model_selection import year_cv_split
from augury.settings import BASE_DIR


FAKE = Faker()
ROW_COUNT = 10
N_FAKE_CATS = 6


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
                self.selector.fit_transform(self.X, pd.Series(dtype="object"))


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
        self.data = (
            CandyStore(seasons=(2017, 2018))
            .match_results()
            .pipe(match.clean_match_data)
            .pipe(common.convert_match_rows_to_teammatch_rows)
        )

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
        self.data = (
            CandyStore(seasons=(2017, 2018))
            .match_results()
            .pipe(match.clean_match_data)
            .pipe(common.convert_match_rows_to_teammatch_rows)
        )
        self.cols_to_drop = ["team", "oppo_score"]
        self.transformer = ColumnDropper(cols_to_drop=self.cols_to_drop)

    def test_transform(self):
        self.transformer.fit(self.data, None)
        transformed_data = self.transformer.transform(self.data)

        for column in self.cols_to_drop:
            self.assertNotIn(column, transformed_data.columns)


class TestDataFrameConverter(TestCase):
    def setUp(self):
        self.data = (
            CandyStore(seasons=(2017, 2018))
            .match_results()
            .pipe(match.clean_match_data)
        )
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


class TestKerasClassifier(TestCase):
    @patch("augury.run.create_pipelines", {"fake": create_fake_pipeline()})
    def setUp(self):
        data = FakeEstimatorData()
        _X_train, _y_train = data.train_data
        self.X_train = _X_train.iloc[:, N_FAKE_CATS:]
        self.y_train = (_y_train > 0).astype(int)

        _X_test, _y_test = data.test_data
        self.X_test = _X_test.iloc[:, N_FAKE_CATS:]

        self.classifier = KerasClassifier(self.model_func, epochs=1)

    def test_predict(self):
        self.classifier.fit(self.X_train, self.y_train)
        predictions = self.classifier.predict(self.X_test)

        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape, (len(self.X_test),))
        self.assertTrue(np.all(np.logical_or(predictions == 0, predictions == 1)))

    def test_predict_proba(self):
        self.classifier.fit(self.X_train, self.y_train)
        predictions = self.classifier.predict_proba(self.X_test)

        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape, (len(self.X_test), 2))
        self.assertTrue(np.all(np.logical_and(predictions >= 0, predictions <= 1)))

    def test_set_params(self):
        self.classifier.set_params(epochs=2)
        self.assertEqual(self.classifier.epochs, 2)

    def test_history(self):
        self.classifier.fit(self.X_train, self.y_train)
        self.assertIsInstance(self.classifier.history, keras.callbacks.History)

    def model_func(self, **_kwargs):
        N_FEATURES = len(self.X_train.columns)

        stats_input = keras.layers.Input(
            shape=(N_FEATURES,), dtype="float32", name="stats"
        )
        layer_n = keras.layers.Dense(10, input_shape=(N_FEATURES,), activation="relu")(
            stats_input
        )
        dropout_n = keras.layers.Dropout(0.1)(layer_n)

        output = keras.layers.Dense(2, activation="softmax")(dropout_n)

        model = keras.models.Model(inputs=stats_input, outputs=output)
        model.compile(loss="categorical_crossentropy", optimizer="adam")

        return lambda: model


class TestSklearn(TestCase, KedroContextMixin):
    def setUp(self):
        self.data = FakeEstimatorData()
        self.estimator = self.load_context().catalog.load("fake_estimator")

    def test_match_accuracy_scorer(self):
        X_test, y_test = self.data.test_data

        match_acc = match_accuracy_scorer(self.estimator, X_test, y_test)

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

    # We use FakeEstimator to generate predictions for #test_bits_scorer,
    # and we don't care if it doesn't converge
    @pytest.mark.filterwarnings("ignore:lbfgs failed to converge")
    def test_bits_scorer(self):
        bits = bits_scorer(self.estimator, *self.data.train_data, proba=False)
        self.assertIsInstance(bits, float)

        with self.subTest("with a superfluous n_years arg"):
            bits_with_year = bits_scorer(
                self.estimator, *self.data.train_data, proba=False, n_years=100
            )
            self.assertEqual(bits, bits_with_year)

        with self.subTest("with an invalid proba arg"):
            with self.assertRaisesRegex(IndexError, "too many indices for array"):
                bits_scorer(self.estimator, *self.data.train_data, proba=True)

        with self.subTest("with a classifier"):
            classifier = LogisticRegression()
            _X_train, y_train = self.data.train_data
            # There are six categorical features in fake data, and we don't need
            # to bother with encoding them
            X_train = _X_train.iloc[:, N_FAKE_CATS:]
            classifier.fit(X_train, y_train)

            class_bits = bits_scorer(classifier, X_train, y_train, proba=True)

            self.assertIsInstance(class_bits, float)

    def test_bits_objective(self):
        y_true = np.random.randint(0, 2, 10)
        y_pred = np.random.random(10)

        grad, hess = bits_objective(y_true, y_pred)

        self.assertIsInstance(grad, np.ndarray)
        self.assertEqual(grad.dtype, "float64")
        self.assertIsInstance(hess, np.ndarray)
        self.assertEqual(hess.dtype, "float64")

        warnings.filterwarnings(
            "error",
            category=RuntimeWarning,
            message="divide by zero encountered in true_divide",
        )

        with self.subTest("when some predictions equal 1"):
            y_pred[:5] = np.ones(5)

            # Will raise a divide-by-zero error if a y_pred value of 1 gets through,
            # so we don't need to assert anything
            bits_objective(y_true, y_pred)

        with self.subTest("when some predictions equal 0"):
            y_pred[:5] = np.zeros(5)

            # Will raise a divide-by-zero error if a y_pred value of 1 gets through,
            # so we don't need to assert anything
            bits_objective(y_true, y_pred)
