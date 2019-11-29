"""Generates predictions with the given models for the given inputs"""

from typing import Tuple, Optional, List
import itertools
from datetime import date

import pandas as pd
import numpy as np
from kedro.context import load_context

from machine_learning.ml_data import MLData
from machine_learning.ml_estimators.base_ml_estimator import BaseMLEstimator
from machine_learning.settings import SEED, ML_MODELS, BASE_DIR

np.random.seed(SEED)


END_OF_YEAR = f"{date.today().year}-12-31"
DEFAULT_ML_MODELS = [ml_model["name"] for ml_model in ML_MODELS]


class Predictor:
    def __init__(
        self,
        pred_year_range: Tuple[int, int],
        round_number: Optional[int] = None,
        verbose: int = 1,
        **data_kwargs,
    ):
        self.pred_year_range = pred_year_range
        self.round_number = round_number
        self.verbose = verbose
        self._context = load_context(
            BASE_DIR,
            start_date=f"{self.pred_year_range[0]}-01-01",
            end_date=f"{self.pred_year_range[1]}-12-31",
            round_number=self.round_number,
        )
        self._data = MLData(**data_kwargs)

    def make_predictions(
        self, ml_model_names: List[str] = DEFAULT_ML_MODELS, train=False
    ) -> pd.DataFrame:
        ml_models = [
            ml_model for ml_model in ML_MODELS if ml_model["name"] in ml_model_names
        ]

        assert any(ml_models), (
            "Couldn't find any ML models, check that at least one "
            f"{ml_model_names} is in ML_MODELS."
        )

        predictions = [
            self._make_predictions_by_year(ml_models, year, train=train)
            for year in range(*self.pred_year_range)
        ]

        return pd.concat(list(itertools.chain.from_iterable(predictions)))

    def _make_predictions_by_year(
        self, ml_models, year: int, train=False
    ) -> List[pd.DataFrame]:
        return [
            self._make_model_predictions(year, ml_model, train=train)
            for ml_model in ml_models
        ]

    def _make_model_predictions(self, year: int, ml_model, train=False) -> pd.DataFrame:
        if self.verbose == 1:
            print(f"Making predictions with {ml_model['name']}")

        loaded_model = self._context.catalog.load(ml_model["name"])
        self._data.data_set = ml_model["data_set"]

        trained_model = self._train_model(loaded_model) if train else loaded_model
        X_test, _ = self._data.test_data()

        assert X_test.any().any(), (
            "X_test doesn't have any rows, likely due to no data being available for "
            f"{year}."
        )

        y_pred = trained_model.predict(X_test)
        data_row_slice = (
            slice(None),
            year,
            slice(self.round_number, self.round_number),
        )

        model_predictions = (
            X_test.assign(predicted_margin=y_pred, ml_model=ml_model["name"])
            .set_index("ml_model", append=True, drop=False)
            .loc[
                data_row_slice,
                [
                    "team",
                    "year",
                    "round_number",
                    "oppo_team",
                    "at_home",
                    "ml_model",
                    "predicted_margin",
                ],
            ]
        )

        assert model_predictions.any().any(), (
            "Model predictions data frame is empty, possibly due to a bad row slice:\n"
            f"{data_row_slice}"
        )

        return model_predictions

    def _train_model(self, ml_model: BaseMLEstimator) -> BaseMLEstimator:
        X_train, y_train = self._data.train_data()

        # On the off chance that we try to run predictions for years that have
        # no relevant prediction data
        assert not X_train.empty and not y_train.empty, (
            "Some required data was missing for training for year range "
            f"{self._data.train_years}.\n"
            f"{'X_train is empty' if X_train.empty else ''}"
            f"{'and ' if X_train.empty and y_train.empty else ''}"
            f"{'y_train is empty' if y_train.empty else ''}"
        )

        ml_model.fit(X_train, y_train)

        return ml_model
