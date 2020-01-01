"""kedro data set based on fetching fresh data from the afl_data service"""

from typing import Any, List, Dict, Callable, Union
import importlib
from datetime import date

import pandas as pd
from kedro.io.core import AbstractDataSet


TODAY = date.today()
JAN = 1
FIRST = 1
DEC = 12
THIRTY_FIRST = 31
# Saved data will generally go to the end of previous year, so default for start_date
# for fetched data is beginning of this year
BEGINNING_OF_YEAR = date(TODAY.year, JAN, FIRST)
END_OF_YEAR = date(TODAY.year, DEC, THIRTY_FIRST)
MODULE_SEPARATOR = "."

DATE_RANGE_TYPE: Dict[str, Dict[str, str]] = {
    "whole_season": {
        "start_date": str(BEGINNING_OF_YEAR),
        "end_date": str(END_OF_YEAR),
    },
    "past_rounds": {"start_date": str(BEGINNING_OF_YEAR), "end_date": str(TODAY)},
    "future_rounds": {"start_date": str(TODAY), "end_date": str(END_OF_YEAR)},
}


class JSONRemoteDataSet(AbstractDataSet):
    """kedro data set based on fetching fresh data from the afl_data service"""

    def __init__(
        self, data_source: Union[Callable, str], date_range_type: str, load_kwargs={},
    ):
        if date_range_type not in DATE_RANGE_TYPE.keys():
            raise ValueError(
                f"Argument date_range_type must be one of {DATE_RANGE_TYPE.keys()}, "
                f"but {date_range_type} was received."
            )

        self._date_range = DATE_RANGE_TYPE[date_range_type]
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
