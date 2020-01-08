"""Classes and functions based on existing Scikit-learn functionality."""

from typing import Sequence, Type, List, Union, Optional, Any, Tuple, Dict
import re
import copy
import math
from functools import partial, update_wrapper
import warnings

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.preprocessing import LabelEncoder
from mypy_extensions import TypedDict
from statsmodels.tsa.base.tsa_model import TimeSeriesModel

from augury.types import R, T
from augury.nodes.base import _validate_required_columns
from augury.nodes import common
from augury.settings import TEAM_NAMES


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

LOG_BASE = 2
MIN_VAL = 1 * 10 ** -10

# Got these default p, d, q values from a combination of statistical tests
# (mostly for d and q) and trial-and-error (mostly for p)
DEFAULT_ORDER = (6, 0, 1)


class AveragingRegressor(_BaseComposition, RegressorMixin):
    """Scikit-Learn-style ensemble regressor for averaging regressors' predictions."""

    def __init__(
        self,
        estimators: Sequence[Tuple[str, BaseEstimator]],
        weights: Optional[List[float]] = None,
    ) -> None:
        """Instantiate an AveragingRegressor object.

        Params
        ------
        estimators: Scikit-learn estimators (and their names) for generating
            base predictions that will be averaged.
        weights: Multipliers for individual base predictions to weight their impact
            on the final prediction.
        """
        super().__init__()

        self.estimators = estimators
        self.weights = weights

        self.__validate_estimators_weights_equality()

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> Type[R]:
        """Fit estimators to the data."""
        self.__validate_estimators_weights_equality()

        for _, estimator in self.estimators:
            estimator.fit(X, y)

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict with each estimator, then average the predictions."""
        self.__validate_estimators_weights_equality()

        predictions = [estimator.predict(X) for _, estimator in self.estimators]

        return np.average(np.array(predictions), axis=0, weights=self.weights)

    # The params Dict is way too complicated to try properly typing it
    def get_params(self, deep=True) -> Dict[str, Any]:
        """Get the params dictionary comprised of all estimators."""
        return super()._get_params("estimators", deep=deep)

    def set_params(self, **params) -> BaseEstimator:
        """Set params on any estimators."""
        super()._set_params("estimators", **params)

        return self

    def __validate_estimators_weights_equality(self):
        if self.weights is not None and len(self.estimators) != len(self.weights):
            raise ValueError(
                f"Received {len(self.estimators)} estimators and {len(self.weights)}"
                "weight values, but they must have the same number."
            )


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """Transformer for filtering out features that are less correlated with labels."""

    def __init__(
        self,
        cols_to_keep: List[str] = [],
        threshold: Optional[float] = None,
        labels=pd.Series(),
    ) -> None:
        """Instantiate a CorrelationSelector transformer.

        Params
        ------
        cols_to_keep: List of feature names to always keep in the data set.
        threshold: Minimum correlation value (exclusive) for keeping a feature.
        labels: Label values from the training data set for calculating
            correlations.
        """
        self.threshold = threshold
        self.labels = labels
        self._cols_to_keep = cols_to_keep
        self._above_threshold_columns = cols_to_keep

    def transform(self, X: pd.DataFrame, _y=None) -> pd.DataFrame:
        """Filter out features with weak correlation with the labels."""
        return X[self._above_threshold_columns]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Type[T]:
        """Calculate feature/label correlations and save high-correlation features."""
        if not any(self.labels) and y is not None:
            self.labels = y

        assert any(
            self.labels
        ), "Need labels argument for calculating feature correlations."

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
        """List columns that never filtered out."""
        return self._cols_to_keep

    @cols_to_keep.setter
    def cols_to_keep(self, cols_to_keep: List[str]) -> None:
        """Set the list of columns to always keep.

        Also resets the overall list of columns to keep to the given list.
        """
        self._cols_to_keep = cols_to_keep
        self._above_threshold_columns = self._cols_to_keep


class EloRegressor(BaseEstimator, RegressorMixin):
    """Elo regression model with a scikit-learn interface."""

    def __init__(
        self,
        k=K,
        x=ELO_X,
        m=M,
        home_ground_advantage=HOME_GROUND_ADVANTAGE,
        s=S,
        season_carryover=SEASON_CARRYOVER,
    ):
        """
        Instantiate an EloRegressor object.

        Params
        ------
        k: Elo model param for regulating for how long match results affect Elo ratings.
        x: Elo model param.
        m: Elo model param.
        home_ground_advantage: Elo model param for how many points an average home team
            is expected to win by.
        s: Elo model param.
        season_carryover: The percentage of a team's end-of-season Elo score
            that is kept for the next season.
        """
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
        """Fit estimator to data."""
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
        """Make predictions.

        Data set used for predictions must follow the training set chronologically.
        Otherwise, an error is raised to avoid making invalid predictions.
        """
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
    """Transformer for converting data frames to be organised by match."""

    def __init__(self, match_cols=MATCH_COLS):
        """
        Instantiate a TeammatchToMatchConverter transformer.

        Params
        ------
        match_cols (list of strings,
            default=["date", "venue", "round_type"]):
            List of match columns that are team neutral (e.g. round_number, venue).
            These won't be renamed with 'home_' or 'away_' prefixes.
        """
        self.match_cols = match_cols
        self._match_cols = list(set(match_cols + MATCH_INDEX_COLS))

    def fit(self, _X, _y=None):
        """Include for consistency with the Scikit-learn interface."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data from being organised by team-match to match.

        This means that input has two rows per match (one row each
        for home and away teams), and output has one row per match (with separate
        columns for home and away team data).
        """
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
    """Transformer that drops named columns from data frames."""

    def __init__(self, cols_to_drop: List[str] = []):
        """Instantiate a ColumbnDropper transformer.

        Params
        ------
        cols_to_drop: List of column names to drop.
        """
        self.cols_to_drop = cols_to_drop

    def fit(self, _X, y=None):  # pylint: disable=unused-argument
        """Include for consistency with Scikit-learn interface."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop the given columns from the data."""
        return X.drop(self.cols_to_drop, axis=1, errors="ignore")


class DataFrameConverter(BaseEstimator, TransformerMixin):
    """Transformer that converts numpy arrays into DataFrames with named axes.

    Resulting data frame is assigned named columns and indices per the initial data sets
    passed to fit/predict. This is mostly for cases when classes from packages
    convert DataFrames to numpy arrays without asking, and later transformers depend
    on named indices/columns to work.
    """

    def __init__(
        self,
        columns: Optional[Union[List[str], pd.Index]] = None,
        index: Optional[Union[List[str], pd.Index]] = None,
    ):
        """Instantiate a DataFrameConverter transformer.

        Params
        ------
        columns: List of column names or a pd.Index to assign as columns.
        index: List of row names or a pd.Index to assign as the index.
        """
        self.columns = columns
        self.index = index

    def fit(self, X, y=None):  # pylint: disable=unused-argument
        """Include for consistency with Scikit-learn interface."""
        return self

    def transform(self, X: Union[pd.DataFrame, np.array]):
        """Convert data into a pandas DataFrame with the given columns and index."""
        if self.columns is not None:
            assert X.shape[1] == len(self.columns), (
                f"X must have the same number of columns {X.shape[1]} "
                f"as self.columns {len(self.columns)}."
            )

        if self.index is not None:
            assert X.shape[0] == len(self.index), (
                f"X must have the same number of rows {X.shape[0]} "
                f"as indicated by self.index {len(self.index)}."
            )

        return pd.DataFrame(X, columns=self.columns, index=self.index)


class TimeSeriesRegressor(BaseEstimator, RegressorMixin):
    """Wrapper class with Scikit-learn API for regressors from statsmodels.tsa."""

    def __init__(
        self,
        stats_model: TimeSeriesModel,
        order: Tuple[int, int, int] = DEFAULT_ORDER,
        exog_cols: List[str] = [],
        **sm_kwargs,
    ):
        """Instantiate a StatsModelsRegressor.

        Params
        ------
        stats_model: A model class from the statsmodels package. So far, has only been
            tested with models from the tsa module.
        order: The `order` param for ARIMA and similar models.
        exog_cols: Names of columns to use as exogeneous variables for ARIMA
            and similar models.
        sm_kwargs: Any other keyword arguments to pass directly to the instantiation
            of the given stats_model.
        """
        self.stats_model = stats_model
        self.order = order
        self.exog_cols = exog_cols
        self.sm_kwargs = sm_kwargs
        self._team_models: Dict[str, TimeSeriesModel] = {}

    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, np.array]):
        """Fit the model to the training data."""
        time_series_df = X.assign(y=y, ts_date=X["date"].dt.date).sort_values(
            "ts_date", ascending=True
        )

        _ = [
            self._fit_team_model(team_name, team_df)
            for team_name, team_df in time_series_df.groupby("team")
        ]

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions."""
        team_predictions = [
            self._predict_with_team_model(X, team_name, team_model)
            for team_name, team_model in self._team_models.items()
        ]

        return pd.concat(team_predictions, axis=0).sort_index()

    def _fit_team_model(self, team_name, team_df):
        n_train_years = team_df["date"].dt.year.drop_duplicates().count()
        y = team_df.set_index("ts_date")["y"]

        # Need smaller p for teams with fewer seasons for training (looking at you GWS),
        # because higher p values raise weird calculation errors deep in statsmodels:
        # "On entry to DLASCL parameter number 4 had an illegal value"
        # The minimum number of years and the hard-coded lower p value are somewhat
        # arbitrary, but were arrived at after a fair bit of trial-and-error.
        p_param = self.order[0] if n_train_years >= 3 else min(4, self.order[0])
        order_param = (p_param, *self.order[1:])

        with warnings.catch_warnings():
            # Match dates are roughly weekly, but since they aren't consistent,
            # trying to coerce weekly frequency doesn't work. Ignoring the specific
            # dates and making up a consistent weekly schedule doesn't improve model
            # performance, so better to just ignore this unhelpful warning.
            warnings.filterwarnings(
                "ignore",
                message=(
                    "A date index has been provided, but it has no associated "
                    "frequency information and so will be ignored when "
                    "e.g. forecasting."
                ),
            )
            self._team_models[team_name] = self.stats_model(
                y, order=order_param, exog=self._exog_arg(team_df), **self.sm_kwargs
            ).fit()

    def _predict_with_team_model(
        self,
        X: pd.DataFrame,
        team_name: str,  # pylint: disable=unused-argument
        team_model: TimeSeriesModel,
    ) -> pd.Series:
        team_df = X.query("team == @team_name").sort_values("date")

        if not team_df.any().any():
            return pd.Series(name="yhat")

        team_df_index = team_df.index

        # TODO: Oversimplification of mapping of X dates onto forecast dates
        # that ignores bye weeks and any other scheduling irregularities,
        # but it's good enough for now.
        forecast, _se, _conf = team_model.forecast(
            steps=len(team_df_index), exog=self._exog_arg(team_df)
        )

        return pd.Series(forecast, name="yhat", index=team_df_index)

    def _exog_arg(self, data_frame: pd.DataFrame) -> Optional[np.array]:
        return data_frame[self.exog_cols].to_numpy() if any(self.exog_cols) else None


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


def _calculate_match_accuracy(X, y_true, y_pred):
    """Scikit-learn metric function for calculating tipping accuracy."""
    team_match_data_frame = X.assign(y_true=y_true, y_pred=y_pred)
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


def create_match_accuracy(X):
    """Return Scikit-learn metric function for calculating tipping accuracy."""
    return update_wrapper(
        partial(_calculate_match_accuracy, X), _calculate_match_accuracy
    )


def match_accuracy_scorer(estimator, X, y):
    """Scikit-learn scorer function for calculating tipping accuracy of an estimator."""
    y_pred = estimator.predict(X)

    return _calculate_match_accuracy(X, y, y_pred)


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
    """Scikit-learn scorer for the bits metric.

    Mostly for use in calls to cross_validate. Calculates a score
    based on the the model's predicted probability of a given result. For this metric,
    higher scores are better.
    """
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
            "year": home_match_data_frame["year"],
        }
    )

    return (
        bits_data_frame.apply(_calculate_bits, axis=1).sum()
        / bits_data_frame["year"].drop_duplicates().count()
    )


def year_cv_split(X, year_range):
    """Split data by year for cross-validation for time-series data.

    Makes data from each year in the year_range a test set per split, with data
    from all earlier years being in the train split.
    """
    return [
        ((X["year"] < year).to_numpy(), (X["year"] == year).to_numpy())
        for year in range(*year_range)
    ]
