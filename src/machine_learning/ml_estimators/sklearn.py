"""Classes based on existing Scikit-learn functionality with slight modifications"""

from typing import Sequence, Type, List, Union, Optional, Any, Tuple
from functools import reduce

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.preprocessing import LabelEncoder
from mypy_extensions import TypedDict

from machine_learning.types import R, T
from machine_learning.nodes.base import _validate_required_columns


EloDictionary = TypedDict(
    "EloDictionary",
    {
        "home_away_elo_ratings": List[Tuple[float, float]],
        "current_team_elo_ratings": np.ndarray,
        "year": int,
    },
)

BASE_RATING = 1000
K = 35.6
ELO_X = 0.49
M = 130
HOME_GROUND_ADVANTAGE = 9
S = 250
SEASON_CARRYOVER = 0.575


class AveragingRegressor(_BaseComposition, RegressorMixin):
    """Scikit-Learn-style ensemble regressor for averaging regressors' predictions"""

    def __init__(
        self,
        estimators: Sequence[Tuple[str, BaseEstimator]],
        weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()

        self.estimators = estimators
        self.weights = weights

        self.__validate_estimators_weights_equality()

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> Type[R]:
        """Fit estimators to data"""

        self.__validate_estimators_weights_equality()

        for _, estimator in self.estimators:
            estimator.fit(X, y)

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict with each estimator, then average the predictions"""

        self.__validate_estimators_weights_equality()

        predictions = [estimator.predict(X) for _, estimator in self.estimators]

        return np.average(np.array(predictions), axis=0, weights=self.weights)

    # The params Dict is way too complicated to try typing it
    def get_params(self, deep=True) -> Any:
        return super()._get_params("estimators", deep=deep)

    def set_params(self, **params) -> BaseEstimator:
        super()._set_params("estimators", **params)

        return self

    def __validate_estimators_weights_equality(self):
        if self.weights is not None and len(self.estimators) != len(self.weights):
            raise ValueError(
                f"Received {len(self.estimators)} estimators and {len(self.weights)}"
                "weight values, but they must have the same number."
            )


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """
    Proprocessing transformer for filtering out features that are less correlated with labels
    """

    def __init__(
        self,
        labels: Optional[pd.Series] = None,
        cols_to_keep: List[str] = [],
        threshold: Optional[float] = None,
    ) -> None:
        self.labels = labels
        self.threshold = threshold
        self._cols_to_keep = cols_to_keep
        self._above_threshold_columns = cols_to_keep

    def transform(self, X: pd.DataFrame, _y=None) -> pd.DataFrame:
        return X[self._above_threshold_columns]

    def fit(self, X: pd.DataFrame, _y=None) -> Type[T]:
        if self.labels is None:
            raise TypeError(
                "Labels for calculating feature correlations haven't been defined."
            )

        data_frame = pd.concat([X, self.labels], axis=1).drop(self.cols_to_keep, axis=1)
        label_correlations = data_frame.corr().fillna(0)[self.labels.name].abs()

        if self.threshold is None:
            correlated_columns = data_frame.columns
        else:
            correlated_columns = data_frame.columns[label_correlations > self.threshold]

        self._above_threshold_columns = self.cols_to_keep + [
            col for col in correlated_columns if col in X.columns
        ]

        return self

    @property
    def cols_to_keep(self) -> List[str]:
        return self._cols_to_keep

    @cols_to_keep.setter
    def cols_to_keep(self, cols_to_keep: List[str]) -> None:
        self._cols_to_keep = cols_to_keep
        self._above_threshold_columns = self._cols_to_keep


class EloRegressor(BaseEstimator, RegressorMixin):
    """Elo regression model with a scikit-learn interface"""

    def __init__(
        self,
        base_rating=BASE_RATING,
        k=K,
        x=ELO_X,
        m=M,
        home_ground_advantage=HOME_GROUND_ADVANTAGE,
        s=S,
        season_carryover=SEASON_CARRYOVER,
    ):
        self.base_rating = base_rating
        self.k = k
        self.x = x
        self.m = m
        self.home_ground_advantage = home_ground_advantage
        self.s = s
        self.season_carryover = season_carryover
        self._team_encoder = LabelEncoder()
        self._elo_ratings = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Type[R]:
        """Fit estimators to data"""

        ELO_INDEX_COLS = {"home_team", "year", "round_number"}
        REQUIRED_COLS = ELO_INDEX_COLS | {
            "home_score",
            "away_score",
            "away_team",
            "date",
        }

        joined_data_frame: pd.DataFrame = pd.concat([X, y], axis=1)

        _validate_required_columns(REQUIRED_COLS, joined_data_frame.columns)

        data_frame = (
            joined_data_frame.set_index(list(ELO_INDEX_COLS), drop=False).rename_axis(
                [None] * len(ELO_INDEX_COLS)
            )
            if ELO_INDEX_COLS != {*joined_data_frame.index.names}
            else joined_data_frame.copy()
        )

        if not data_frame.index.is_monotonic:
            data_frame.sort_index(inplace=True)

        self.team_encoder.fit(data_frame["home_team"])
        time_sorted_data_frame = data_frame.sort_values(
            ["year", "round_number"], ascending=True
        )

        elo_matrix = (
            time_sorted_data_frame.assign(
                home_team=lambda df: self.team_encoder.transform(df["home_team"]),
                away_team=lambda df: self.team_encoder.transform(df["away_team"]),
            )
            .eval("home_margin = home_score - away_score")
            .loc[:, ["year", "home_team", "away_team", "home_margin"]]
        ).to_numpy()

        current_team_elo_ratings = np.full(
            len(set(data_frame["home_team"])), self.base_rating
        )

        starting_elo_dictionary: EloDictionary = {
            "home_away_elo_ratings": [],
            "current_team_elo_ratings": current_team_elo_ratings,
            "year": 0,
        }

        self._elo_ratings = self._calculate_elo_ratings(
            elo_matrix, starting_elo_dictionary, time_sorted_data_frame.index
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with each estimator, then average the predictions"""

        max_year_with_ratings = self._elo_ratings.index.get_level_values(1).max()
        max_round_with_ratings = (
            self._elo_ratings.loc[(slice(None), max_year_with_ratings, slice(None))]
            .index.get_level_values(2)
            .max()
        )

        time_sorted_data_frame = X.groupby("home_team")[["year", "round_number"]].max()

        elo_matrix = (
            time_sorted_data_frame.assign(
                home_team=lambda df: self.team_encoder.transform(df["home_team"]),
                away_team=lambda df: self.team_encoder.transform(df["away_team"]),
            )
            .eval("home_margin = home_score - away_score")
            .loc[:, ["year", "home_team", "away_team", "home_margin"]]
        ).to_numpy()

        current_team_elo_ratings = np.full(
            len(set(data_frame["home_team"])), self.base_rating
        )

        starting_elo_dictionary: EloDictionary = {
            "home_away_elo_ratings": [],
            "current_team_elo_ratings": current_team_elo_ratings,
            "year": 0,
        }

        self._elo_ratings = self._calculate_elo_ratings(
            elo_matrix, starting_elo_dictionary, time_sorted_data_frame.index
        )

        return np.average(np.array(predictions), axis=0, weights=self.weights)

    # Basing Elo calculations on:
    # http://www.matterofstats.com/mafl-stats-journal/2013/10/13/building-your-own-team-rating-system.html
    def _elo_formula(
        self,
        prev_elo_rating: float,
        prev_oppo_elo_rating: float,
        margin: int,
        at_home: bool,
    ) -> float:
        home_ground_advantage = (
            self.home_ground_advantage if at_home else self.home_ground_advantage * -1
        )
        expected_outcome = 1 / (
            1
            + 10
            ** (
                (prev_oppo_elo_rating - prev_elo_rating - home_ground_advantage)
                / self.s
            )
        )
        actual_outcome = self.x + 0.5 - self.x ** (1 + (margin / M))

        return prev_elo_rating + (self.k * (actual_outcome - expected_outcome))

    # Assumes df sorted by year & round_number with ascending=True in order to calculate
    # correct Elo ratings
    def _calculate_match_elo_rating(
        self,
        elo_ratings: EloDictionary,
        # match_row = [year, home_team, away_team, home_margin]
        match_row: np.ndarray,
    ) -> EloDictionary:
        match_year = match_row[0]

        # It's typical for Elo models to do a small adjustment toward the baseline between
        # seasons
        if match_year != elo_ratings["year"]:
            prematch_team_elo_ratings = (
                elo_ratings["current_team_elo_ratings"] * self.season_carryover
            ) + self.base_rating * (1 - self.season_carryover)
        else:
            prematch_team_elo_ratings = elo_ratings["current_team_elo_ratings"].copy()

        home_team = int(match_row[1])
        away_team = int(match_row[2])
        home_margin = match_row[3]

        prematch_home_elo_rating = prematch_team_elo_ratings[home_team]
        prematch_away_elo_rating = prematch_team_elo_ratings[away_team]

        home_elo_rating = self._elo_formula(
            prematch_home_elo_rating, prematch_away_elo_rating, home_margin, True
        )
        away_elo_rating = self._elo_formula(
            prematch_away_elo_rating, prematch_home_elo_rating, home_margin * -1, False
        )

        postmatch_team_elo_ratings = prematch_team_elo_ratings.copy()
        postmatch_team_elo_ratings[home_team] = home_elo_rating
        postmatch_team_elo_ratings[away_team] = away_elo_rating

        return {
            "home_away_elo_ratings": elo_ratings["home_away_elo_ratings"]
            + [(prematch_home_elo_rating, prematch_away_elo_rating)],
            "current_team_elo_ratings": postmatch_team_elo_ratings,
            "year": match_year,
        }

    def _calculate_elo_ratings(
        self,
        elo_matrix: np.ndarray,
        elo_dictionary: EloDictionary,
        elo_index: pd.MultiIndex,
    ) -> pd.DataFrame:
        """Add Elo rating of team prior to matches"""

        elo_columns = reduce(
            self._calculate_match_elo_rating, elo_matrix, elo_dictionary
        )["home_away_elo_ratings"]

        return pd.DataFrame(
            elo_columns,
            columns=["home_elo_rating", "away_elo_rating"],
            index=elo_index,
        ).sort_index()
