"""
Classes based on existing Scikit-learn classes or functionality with slight
modifications
"""

from typing import Sequence, Type, List, Union, Optional, Any, Tuple
import re
import copy
import math

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.preprocessing import LabelEncoder
from mypy_extensions import TypedDict

from machine_learning.types import R, T
from machine_learning.nodes.base import _validate_required_columns
from machine_learning.nodes import common
from machine_learning.settings import TEAM_NAMES


EloDictionary = TypedDict(
    "EloDictionary",
    {
        "prev_team_elo_ratings": np.ndarray,
        "current_team_elo_ratings": np.ndarray,
        "year": int,
        "round_number": int,
    },
)


MATCH_INDEX_COLS = ["year", "round_number"]
MATCH_COLS = ["date", "venue", "round_type"]
OPPO_REGEX = re.compile("^oppo_")

ELO_INDEX_COLS = ["home_team", "year", "round_number"]
NULL_TEAM_NAME = "0"

BASE_RATING = 1000
K = 35.6
ELO_X = 0.49
M = 130
HOME_GROUND_ADVANTAGE = 9
S = 250
SEASON_CARRYOVER = 0.575

MATRIX_COLS = [
    "year",
    "round_number",
    "home_team",
    "home_prev_match_at_home",
    "home_prev_match_oppo_team",
    "home_prev_match_margin",
    "away_team",
    "away_prev_match_at_home",
    "away_prev_match_oppo_team",
    "away_prev_match_margin",
]
YEAR_IDX = MATRIX_COLS.index("year")
ROUND_NUMBER_IDX = MATRIX_COLS.index("round_number")
HOME_TEAM_IDX = MATRIX_COLS.index("home_team")
AWAY_TEAM_IDX = MATRIX_COLS.index("away_team")
PREV_AT_HOME_OFFSET = 1
PREV_OPPO_OFFSET = 2
PREV_MARGIN_OFFSET = 3

YEAR_LVL = 1
ROUND_NUMBER_LVL = 2


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
        self, cols_to_keep: List[str] = [], threshold: Optional[float] = None
    ) -> None:
        self.threshold = threshold
        self._labels = pd.Series()
        self._cols_to_keep = cols_to_keep
        self._above_threshold_columns = cols_to_keep

    def transform(self, X: pd.DataFrame, _y=None) -> pd.DataFrame:
        return X[self._above_threshold_columns]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Type[T]:
        if not any(self._labels):
            self._labels = y

        assert any(
            self._labels
        ), "Need labels argument for calculating feature correlations."

        data_frame = pd.concat([X, self._labels], axis=1).drop(
            self.cols_to_keep, axis=1
        )
        label_correlations = data_frame.corr().fillna(0)[self._labels.name].abs()

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
        k=K,
        x=ELO_X,
        m=M,
        home_ground_advantage=HOME_GROUND_ADVANTAGE,
        s=S,
        season_carryover=SEASON_CARRYOVER,
    ):
        self.k = k
        self.x = x
        self.m = m
        self.home_ground_advantage = home_ground_advantage
        self.s = s
        self.season_carryover = season_carryover
        self._running_elo_ratings = {
            "previous_elo": np.array([]),
            "current_elo": np.array([]),
            "year": 0,
            "round_number": 0,
        }
        self._fitted_elo_ratings = copy.deepcopy(self._running_elo_ratings)
        self._first_fitted_year = 0

        self._team_encoder = LabelEncoder()
        # Have to fit encoder on all team names to not make it dependent
        # on the teams in the train set being a superset of those in the test set.
        # Have to add '0' to team names to account for filling in prev_match_oppo_team
        # for a new team's first match
        self._team_encoder.fit(np.append(np.array(TEAM_NAMES), NULL_TEAM_NAME))
        self._null_team = self._team_encoder.transform([NULL_TEAM_NAME])[0]

    def fit(self, X: pd.DataFrame, _y: pd.Series = None) -> Type[R]:
        """Fit estimators to data"""

        REQUIRED_COLS = set(ELO_INDEX_COLS) | set(MATRIX_COLS)
        _validate_required_columns(REQUIRED_COLS, X.columns)

        data_frame: pd.DataFrame = (
            X.set_index(ELO_INDEX_COLS, drop=False)
            .rename_axis([None] * len(ELO_INDEX_COLS))
            .assign(
                home_team=lambda df: self._team_encoder.transform(df["home_team"]),
                away_team=lambda df: self._team_encoder.transform(df["away_team"]),
                home_prev_match_oppo_team=lambda df: self._team_encoder.transform(
                    df["home_prev_match_oppo_team"].astype(str)
                ),
                away_prev_match_oppo_team=lambda df: self._team_encoder.transform(
                    df["away_prev_match_oppo_team"].astype(str)
                ),
            )
            .sort_index(level=[YEAR_LVL, ROUND_NUMBER_LVL], ascending=True)
        )

        self._reset_elo_state()

        elo_matrix = (data_frame.loc[:, MATRIX_COLS]).to_numpy()

        for match_row in elo_matrix:
            self._update_current_elo_ratings(match_row)

        self._fitted_elo_ratings = copy.deepcopy(self._running_elo_ratings)
        self._first_fitted_year = data_frame["year"].min()

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with each estimator, then average the predictions"""

        REQUIRED_COLS = set(ELO_INDEX_COLS) | {"away_team"}

        _validate_required_columns(REQUIRED_COLS, X.columns)

        assert X.index.is_monotonic, (
            "Test data must be sorted by index before passing it to the model, ",
            "because calculating Elo ratings requires multiple resorts, ",
            "and we need to be able to make the final predictions match the sorting ",
            "of the input data.",
        )

        data_frame = (
            X.assign(
                home_team=lambda df: self._team_encoder.transform(df["home_team"]),
                away_team=lambda df: self._team_encoder.transform(df["away_team"]),
                home_prev_match_oppo_team=lambda df: self._team_encoder.transform(
                    df["home_prev_match_oppo_team"].astype(str)
                ),
                away_prev_match_oppo_team=lambda df: self._team_encoder.transform(
                    df["away_prev_match_oppo_team"].astype(str)
                ),
            )
            .set_index(ELO_INDEX_COLS, drop=False)
            .sort_index(level=[YEAR_LVL, ROUND_NUMBER_LVL], ascending=True)
        )

        elo_matrix = (data_frame.loc[:, MATRIX_COLS]).to_numpy()

        if data_frame["year"].min() == self._first_fitted_year:
            self._reset_elo_state()
        else:
            self._running_elo_ratings = copy.deepcopy(self._fitted_elo_ratings)

        elo_predictions = [
            self._calculate_current_elo_predictions(match_row)
            for match_row in elo_matrix
        ]
        # Need to zip predictions to group by home/away instead of match,
        # then transpose to convert from rows to columns
        home_away_columns = np.array(list(zip(*elo_predictions))).T

        elo_data_frame = pd.DataFrame(
            home_away_columns,
            columns=["home_elo_prediction", "away_elo_prediction"],
            index=data_frame.index,
        ).sort_index()

        match_data_frame = pd.concat(
            [elo_data_frame, data_frame.sort_index().loc[:, ["away_team", "date"]]],
            axis=1,
        ).reset_index(drop=False)

        # Have to convert to team-match rows to make shape consistent with other
        # model predictions
        return (
            common.convert_match_rows_to_teammatch_rows(match_data_frame)
            # Need to make sure sorting matches wider data set conventions
            .sort_index()
            .loc[:, "elo_prediction"]
            .to_numpy()
        )

    def _calculate_current_elo_predictions(self, match_row: np.ndarray):
        home_team = int(match_row[HOME_TEAM_IDX])
        away_team = int(match_row[AWAY_TEAM_IDX])

        self._update_current_elo_ratings(match_row)

        home_elo_rating = self._running_elo_ratings["current_elo"][home_team]
        away_elo_rating = self._running_elo_ratings["current_elo"][away_team]

        home_elo_prediction = self._calculate_team_elo_prediction(
            home_elo_rating, away_elo_rating, True
        )
        away_elo_prediction = self._calculate_team_elo_prediction(
            away_elo_rating, home_elo_rating, False
        )

        return home_elo_prediction, away_elo_prediction

    # Assumes df sorted by year & round_number with ascending=True in order to calculate
    # correct Elo ratings
    def _update_current_elo_ratings(self, match_row: np.ndarray) -> None:
        home_team = int(match_row[HOME_TEAM_IDX])
        away_team = int(match_row[AWAY_TEAM_IDX])

        self._update_prev_elo_ratings(match_row)

        home_elo_rating = self._calculate_current_elo_rating(HOME_TEAM_IDX, match_row)
        away_elo_rating = self._calculate_current_elo_rating(AWAY_TEAM_IDX, match_row)

        self._running_elo_ratings["current_elo"][home_team] = home_elo_rating
        self._running_elo_ratings["current_elo"][away_team] = away_elo_rating

    def _update_prev_elo_ratings(self, match_row: np.ndarray):
        match_year = match_row[YEAR_IDX]
        match_round = match_row[ROUND_NUMBER_IDX]

        self._validate_consecutive_rounds(match_year, match_round)

        # Need to wait till new round to update prev_team_elo_ratings to avoid
        # updating an Elo rating from the previous round before calculating
        # a relevant team's Elo for the current round
        if match_round != self._running_elo_ratings["round_number"]:
            self._running_elo_ratings["previous_elo"] = np.copy(
                self._running_elo_ratings["current_elo"]
            )

            self._running_elo_ratings["round_number"] = match_round

        # It's typical for Elo models to do a small adjustment toward the baseline
        # between seasons
        if match_year != self._running_elo_ratings["year"]:
            self._running_elo_ratings["previous_elo"] = (
                self._running_elo_ratings["previous_elo"] * self.season_carryover
            ) + BASE_RATING * (1 - self.season_carryover)

            self._running_elo_ratings["year"] = match_year

    def _calculate_current_elo_rating(self, team_idx: int, match_row: np.ndarray):
        team = int(match_row[team_idx])
        was_at_home = int(match_row[team_idx + PREV_AT_HOME_OFFSET])
        prev_oppo_team = int(match_row[team_idx + PREV_OPPO_OFFSET])
        prev_margin = match_row[team_idx + PREV_MARGIN_OFFSET]

        # If a previous oppo team has the null team value, that means this is
        # the given team's first match, so they start with the default Elo rating
        if prev_oppo_team == self._null_team:
            return self._running_elo_ratings["previous_elo"][team]

        prev_elo_rating = self._running_elo_ratings["previous_elo"][team]
        prev_oppo_elo_rating = self._running_elo_ratings["previous_elo"][prev_oppo_team]
        prev_elo_prediction = self._calculate_team_elo_prediction(
            prev_elo_rating, prev_oppo_elo_rating, was_at_home
        )

        return self._calculate_team_elo_rating(
            prev_elo_rating, prev_elo_prediction, prev_margin
        )

    # Basing Elo calculations on:
    # http://www.matterofstats.com/mafl-stats-journal/2013/10/13/building-your-own-team-rating-system.html
    def _calculate_team_elo_prediction(
        self, elo_rating: int, oppo_elo_rating: int, at_home: int
    ) -> float:
        home_ground_advantage = (
            self.home_ground_advantage
            if at_home == 1
            else self.home_ground_advantage * -1
        )

        return 1 / (
            1 + 10 ** ((oppo_elo_rating - elo_rating - home_ground_advantage) / self.s)
        )

    def _calculate_team_elo_rating(
        self, elo_rating: float, elo_prediction: float, margin: int
    ) -> float:
        actual_outcome = self.x + 0.5 - self.x ** (1 + (margin / M))

        return elo_rating + (self.k * (actual_outcome - elo_prediction))

    def _reset_elo_state(self):
        self._running_elo_ratings["previous_elo"] = np.full(
            len(TEAM_NAMES) + 1, BASE_RATING
        )
        self._running_elo_ratings["current_elo"] = np.full(
            len(TEAM_NAMES) + 1, BASE_RATING
        )
        self._running_elo_ratings["year"] = 0
        self._running_elo_ratings["round_number"] = 0

    def _validate_consecutive_rounds(self, match_year: int, match_round: int):
        is_start_of_data = (
            self._running_elo_ratings["year"] == 0
            and self._running_elo_ratings["round_number"] == 0
        )
        is_new_year = (
            match_year - self._running_elo_ratings["year"] == 1 and match_round == 1
        )
        is_same_round = match_round == self._running_elo_ratings["round_number"]
        is_next_round = match_round - self._running_elo_ratings["round_number"] == 1

        assert is_start_of_data or is_new_year or is_same_round or is_next_round, (
            "For Elo calculations to be valid, we must update ratings for each round. "
            f"The current year/round of {match_year}/{match_round} seems to skip "
            f"the last-calculated year/round of {self._running_elo_ratings['year']}/"
            f"{self._running_elo_ratings['round_number']}"
        )


class TeammatchToMatchConverter(BaseEstimator, TransformerMixin):
    """
    Transformer for converting data frames from having one team-match combination per
    row to one match per row.

    Parameters:
        match_cols (list of strings,
            default=["date", "venue", "round_type"]):
            List of match columns that are team neutral (e.g. round_number, venue).
            These won't be renamed with 'home_' or 'away_' prefixes.
    """

    def __init__(self, match_cols=MATCH_COLS):
        self.match_cols = match_cols
        self._match_cols = list(set(match_cols + MATCH_INDEX_COLS))

    def fit(self, _X, _y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_required_columns(X)

        return (
            pd.concat(
                [
                    self._match_data_frame(X, at_home=True),
                    self._match_data_frame(X, at_home=False),
                ],
                axis=1,
            )
            .reset_index()
            .set_index(["home_team"] + MATCH_INDEX_COLS, drop=False)
            .rename_axis([None] * (len(MATCH_INDEX_COLS) + 1))
            .sort_index()
        )

    def _validate_required_columns(self, data_frame: pd.DataFrame):
        required_cols: List[str] = ["team", "oppo_team", "at_home"] + self._match_cols

        _validate_required_columns(required_cols, data_frame.columns)

    def _match_data_frame(
        self, data_frame: pd.DataFrame, at_home: bool = True
    ) -> pd.DataFrame:
        home_index = "team" if at_home else "oppo_team"
        away_index = "oppo_team" if at_home else "team"
        # We drop oppo stats cols, because we end up with both teams' stats per match
        # when we join home and away teams. We keep 'oppo_team' and add the renamed column
        # to the index for convenience
        oppo_stats_cols = [
            col
            for col in data_frame.columns
            if re.match(OPPO_REGEX, col) and col != "oppo_team"
        ]

        return (
            data_frame.query(f"at_home == {int(at_home)}")
            # We index match rows by home_team, year, round_number
            .rename(columns={home_index: "home_team", away_index: "away_team"})
            .drop(["at_home"] + oppo_stats_cols, axis=1)
            # We add all match cols to the index, because they don't affect the upcoming
            # concat, and it's easier than creating a third data frame for match cols
            .set_index(["home_team", "away_team"] + self._match_cols)
            .rename(columns=self._replace_col_names(at_home))
            .sort_index()
        )

    @staticmethod
    def _replace_col_names(at_home: bool):
        team_label = "home" if at_home else "away"
        oppo_label = "away" if at_home else "home"

        return (
            lambda col: col.replace("oppo_", f"{oppo_label}_", 1)
            if re.match(OPPO_REGEX, col)
            else f"{team_label}_{col}"
        )


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Transformer that drops named columns from data frames.
    """

    def __init__(self, cols_to_drop: List[str] = []):
        self.cols_to_drop = cols_to_drop

    def fit(self, _X, _y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(self.cols_to_drop, axis=1, errors="ignore")


def _calculate_team_margin(team_margin, oppo_margin):
    # We want True to be 1 and False to be -1
    team_margin_multiplier = ((team_margin > oppo_margin).astype(int) * 2) - 1

    return (
        pd.Series(
            ((team_margin.abs() + oppo_margin.abs()) / 2) * team_margin_multiplier
        )
        .reindex(team_margin.index)
        .sort_index()
    )


def match_accuracy_scorer(estimator, X, y):
    """Scikit-learn scorer function for calculating tipping accuracy of an estimator"""

    y_pred = estimator.predict(X)

    team_match_data_frame = X.assign(y_true=y, y_pred=y_pred)
    home_match_data_frame = team_match_data_frame.query("at_home == 1").sort_index()
    away_match_data_frame = (
        team_match_data_frame.query("at_home == 0")
        .set_index(["oppo_team", "year", "round_number"])
        .rename_axis([None, None, None])
        .sort_index()
    )

    home_margin = _calculate_team_margin(
        home_match_data_frame["y_true"], away_match_data_frame["y_true"]
    )
    home_pred_margin = _calculate_team_margin(
        home_match_data_frame["y_pred"], away_match_data_frame["y_pred"]
    )

    return (
        # Any zero margin (i.e. a draw) is counted as correct per usual tipping rules.
        # Predicted margins should never be zero, but since we don't want to encourage
        # any wayward models, we'll count a predicted margin of zero as incorrect
        ((home_margin >= 0) & (home_pred_margin > 0))
        | ((home_margin <= 0) & (home_pred_margin < 0))
    ).mean()


LOG_BASE = 2
MIN_VAL = 1 * 10 ** -10


def _calculate_bits(row):
    if row["home_pred"] > row["away_pred"]:
        predicted_win_proba = row["home_pred"]
        predicted_home_win = True
    else:
        predicted_win_proba = row["away_pred"]
        predicted_home_win = False

    correct = (predicted_home_win and row["home_win"]) or (
        not predicted_home_win and not row["home_win"]
    )

    if row["draw"]:
        return 1 + (
            0.5
            * math.log(
                max(predicted_win_proba * (1 - predicted_win_proba), MIN_VAL), LOG_BASE
            )
        )

    if correct:
        return 1 + math.log(max(row["home_pred"], MIN_VAL), LOG_BASE)

    return 1 + math.log(max(1 - predicted_win_proba, MIN_VAL), LOG_BASE)


def bits_scorer(estimator, X, y):
    y_pred = estimator.predict(X)

    team_match_data_frame = X.assign(y_true=y.to_numpy(), y_pred=y_pred)
    home_match_data_frame = team_match_data_frame.query("at_home == 1").sort_index()
    away_match_data_frame = (
        team_match_data_frame.query("at_home == 0")
        .set_index(["oppo_team", "year", "round_number"])
        .rename_axis([None, None, None])
        .sort_index()
    )

    bits_data_frame = pd.DataFrame(
        {
            "home_win": home_match_data_frame["y_true"]
            > away_match_data_frame["y_true"],
            "draw": home_match_data_frame["y_true"] == away_match_data_frame["y_true"],
            "home_pred": home_match_data_frame["y_pred"],
            "away_pred": away_match_data_frame["y_pred"],
        }
    )

    return bits_data_frame.apply(_calculate_bits, axis=1).sum()


def year_cv_split(X, year_range):
    """Split data by year for cross-validation for time-series data"""

    return [
        ((X["year"] < year).to_numpy(), (X["year"] == year).to_numpy())
        for year in range(*year_range)
    ]
