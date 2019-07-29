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


class JSONRemoteDataSet(AbstractDataSet):
    """kedro data set based on fetching fresh data from the afl_data service"""

    def __init__(
        self,
        data_source: Union[Callable, str],
        start_date: str = str(BEGINNING_OF_YEAR),
        end_date: str = str(END_OF_YEAR),
    ):
        self.start_date = start_date
        self.end_date = end_date

        if callable(data_source):
            self.data_source = data_source
        else:
            path_parts = data_source.split(MODULE_SEPARATOR)
            function_name = path_parts[-1]
            module_path = MODULE_SEPARATOR.join(path_parts[:-1])
            module = importlib.import_module(module_path)

            self.data_source = getattr(module, function_name)

    def _load(self) -> List[Dict[str, Any]]:
        return self.data_source(start_date=self.start_date, end_date=self.end_date)

    def _save(self, data: pd.DataFrame) -> None:
        pass

    def _describe(self):
        return {"start_date": self.start_date, "end_date": self.end_date}
