"""kedro data set based on fetching fresh data from the afl_data service."""

from typing import Any, List, Dict, Callable, Union, Optional
import importlib
from datetime import date, timedelta

import pandas as pd
from kedro.io.core import AbstractDataSet


TODAY = date.today()
JAN = 1
FIRST = 1
DEC = 12
THIRTY_FIRST = 31
WEEK = 7
# Saved data will generally go to the end of previous year, so default for start_date
# for fetched data is beginning of this year
BEGINNING_OF_YEAR = date(TODAY.year, JAN, FIRST)
END_OF_YEAR = date(TODAY.year, DEC, THIRTY_FIRST)
MODULE_SEPARATOR = "."
# We make a week ago the dividing line between past & future rounds,
# because "past" data sets aren't updated until a few days after a given round is over,
# meaning we have to keep relying on "future" data sets for past matches
# if we run the pipeline mid-round
START_OF_WEEK = TODAY - timedelta(days=TODAY.weekday())
ONE_WEEK_AGO = TODAY - timedelta(days=WEEK)

DATE_RANGE_TYPE: Dict[str, Dict[str, str]] = {
    "whole_season": {
        "start_date": str(BEGINNING_OF_YEAR),
        "end_date": str(END_OF_YEAR),
    },
    "past_rounds": {
        "start_date": str(BEGINNING_OF_YEAR),
        "end_date": str(START_OF_WEEK),
    },
    "future_rounds": {"start_date": str(ONE_WEEK_AGO), "end_date": str(END_OF_YEAR)},
}


class JSONRemoteDataSet(AbstractDataSet):
    """Kedro data set based on fetching fresh data from the afl_data service."""

    def __init__(
        self,
        data_source: Union[Callable, str],
        date_range_type: Optional[str] = None,
        **load_kwargs,
    ):
        """Instantiate a JSONRemoteDataSet object.

        Params
        ------
        data_source: Either a function that fetches data from an external API,
            or a reference to one that can be loaded via `getattr`.
        date_range_type: Defines the date range of the data to be fetched.
            Can be one of the following:
                'whole_season': all of the current year.
                'past_rounds': the current year up to the current date (inclusive).
                'future_rounds': the current date until the end of the current year
                    (inclusive).
        load_kwargs: Keyword arguments to pass to the data import function.
        """
        self._validate_date_range_type(date_range_type)

        self._date_range = (
            {} if date_range_type is None else DATE_RANGE_TYPE[date_range_type]
        )
        self._data_source_kwargs: Dict[str, Any] = {**self._date_range, **load_kwargs}

        if callable(data_source):
            self.data_source = data_source
        else:
            path_parts = data_source.split(MODULE_SEPARATOR)
            function_name = path_parts[-1]
            module_path = MODULE_SEPARATOR.join(path_parts[:-1])
            module = importlib.import_module(module_path)

            self.data_source = getattr(module, function_name)

    def _load(self) -> List[Dict[str, Any]]:
        return self.data_source(**self._data_source_kwargs)

    def _save(self, data: pd.DataFrame) -> None:
        pass

    def _describe(self):
        return self._data_source_kwargs

    @staticmethod
    def _validate_date_range_type(date_range_type: Optional[str]) -> None:
        assert date_range_type is None or date_range_type in DATE_RANGE_TYPE, (
            "Argument date_range_type must be None or one of "
            f"{DATE_RANGE_TYPE.keys()}, but {date_range_type} was received."
        )
