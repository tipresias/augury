"""Generates predictions with the given models for the given inputs."""

from typing import List, Optional, Dict
import itertools

import pandas as pd
import numpy as np
from kedro.framework.context import KedroContext

from augury.ml_data import MLData
from augury.ml_estimators.base_ml_estimator import BaseMLEstimator
from augury.types import YearRange, MLModelDict
from augury.settings import SEED, PREDICTION_TYPES

np.random.seed(SEED)


PREDICTION_COLS = [
    "team",
    "year",
    "round_number",
    "oppo_team",
    "at_home",
    "ml_model",
] + [f"predicted_{pred_type}" for pred_type in PREDICTION_TYPES]


class Predictor:
    """Generates predictions with the given models for the given inputs."""

    def __init__(
        self,
        year_range: YearRange,
        context: "augury.settings.ProjectContext",
        round_number: Optional[int] = None,
        train=False,
        verbose: int = 1,
        **data_kwargs,
    ):
        """Instantiate Predictor object.

        Params
        ------
        year_range: Year range for which to make predictions (first year inclusive,
            last year exclusive, per `range` function).
        context: Kedro context object as defined in settings.ProjectContext.
        round_number: Round number for which to make predictions. If omitted,
            predictions are made for entire seasons.
        train: Whether to train each model on data from previous years
            before making predictions on a given year's matches.
        verbose: (1 or 0) Whether to print information while making predictions.
        data_kwargs: Keyword arguments to pass to MLData on instantiation.
        """
        self.context = context
        self.year_range = year_range
        self.round_number = round_number
        self.train = train
        self.verbose = verbose
        self._data = MLData(
            context=context,
            train_year_range=(min(year_range),),
            test_year_range=year_range,
            **data_kwargs,
        )

    def make_predictions(self, ml_models: List[MLModelDict]) -> pd.DataFrame:
        """Predict margins or confidence percentages for matches."""
        assert any(ml_models), "No ML model info was given."

        predictions = [
            self._make_predictions_by_year(ml_models, year)
            for year in range(*self.year_range)
        ]

        if self.verbose == 1:
            print("Finished making predictions!")

        return pd.concat(list(itertools.chain.from_iterable(predictions)), sort=False)

    def _make_predictions_by_year(self, ml_models, year: int) -> List[pd.DataFrame]:
        return [self._make_model_predictions(year, ml_model) for ml_model in ml_models]

    def _make_model_predictions(self, year: int, ml_model) -> pd.DataFrame:
        if self.verbose == 1:
            print(f"Making predictions with {ml_model['name']}")

        loaded_model = self.context.catalog.load(ml_model["name"])
        self._data.data_set = ml_model["data_set"]
        self._data.label_col = ml_model["label_col"]

        trained_model = self._train_model(loaded_model) if self.train else loaded_model
        X_test, _ = self._data.test_data

        assert X_test.any().any(), (
            "X_test doesn't have any rows, likely due to no data being available for "
            f"{year}."
        )

        y_pred = trained_model.predict(X_test)

        assert not any(np.isnan(y_pred)), (
            f"Predictions should never be NaN, but {trained_model.name} predicted:\n"
            f"{y_pred}."
        )

        data_row_slice = (
            slice(None),
            year,
            slice(self.round_number, self.round_number),
        )

        model_predictions = (
            X_test.assign(**self._prediction_data(ml_model, y_pred))
            .set_index("ml_model", append=True, drop=False)
            .loc[data_row_slice, PREDICTION_COLS]
        )

        assert model_predictions.any().any(), (
            "Model predictions data frame is empty, possibly due to a bad row slice:\n"
            f"{data_row_slice}"
        )

        return model_predictions

    def _train_model(self, ml_model: BaseMLEstimator) -> BaseMLEstimator:
        assert max(self._data.train_year_range) <= min(self._data.test_year_range)

        X_train, y_train = self._data.train_data

        # On the off chance that we try to run predictions for years that have
        # no relevant prediction data
        assert not X_train.empty and not y_train.empty, (
            "Some required data was missing for training for year range "
            f"{self._data.train_year_range}.\n"
            f"{'X_train is empty' if X_train.empty else ''}"
            f"{'and ' if X_train.empty and y_train.empty else ''}"
            f"{'y_train is empty' if y_train.empty else ''}"
        )

        ml_model.fit(X_train, y_train)

        return ml_model

    @staticmethod
    def _prediction_data(
        ml_model: MLModelDict, y_pred: np.ndarray
    ) -> Dict[str, Optional[np.ndarray]]:
        model_pred_type = ml_model["prediction_type"]

        return {
            **{
                f"predicted_{pred_type}": np.nan
                for pred_type in PREDICTION_TYPES
                if pred_type != model_pred_type
            },
            f"predicted_{model_pred_type}": y_pred,
            "ml_model": ml_model["name"],
        }
