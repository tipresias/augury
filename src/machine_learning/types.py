"""Module for custom static data types"""

from typing import Callable, Tuple, Optional, TypeVar, Dict, Any, List, Sequence, Union
from datetime import datetime

from mypy_extensions import TypedDict
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

DataFrameTransformer = Callable[[Union[pd.DataFrame, List[pd.DataFrame]]], pd.DataFrame]
YearPair = Tuple[Optional[int], Optional[int]]
DataReadersParam = Dict[str, Tuple[Callable, Dict[str, Any]]]

DataFrameCalculator = Callable[[pd.DataFrame], pd.Series]
Calculator = Callable[[Sequence[str]], DataFrameCalculator]
CalculatorPair = Tuple[Calculator, List[Sequence[str]]]

R = TypeVar("R", BaseEstimator, RegressorMixin)
T = TypeVar("T", BaseEstimator, TransformerMixin)

BettingData = TypedDict(
    "BettingData",
    {
        "date": datetime,
        "season": int,
        "round_number": int,
        "round": str,
        "home_team": str,
        "away_team": str,
        "home_score": int,
        "away_score": int,
        "home_margin": int,
        "away_margin": int,
        "home_win_odds": float,
        "away_win_odds": float,
        "home_win_paid": float,
        "away_win_paid": float,
        "home_line_odds": float,
        "away_line_odds": float,
        "home_line_paid": float,
        "away_line_paid": float,
        "venue": str,
    },
)
