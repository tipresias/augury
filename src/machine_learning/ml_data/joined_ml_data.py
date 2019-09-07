"""Module for machine learning data class that joins various data sources together"""

from typing import Callable, Optional
from datetime import date

import pandas as pd

from machine_learning.types import YearPair
from machine_learning.run import run_pipeline
from . import BaseMLData


END_OF_YEAR = f"{date.today().year}-12-31"


class JoinedMLData(BaseMLData):
    """Load and clean data from all data sources"""

    def __init__(
        self,
        pipeline_runner: Callable[
            [str, str, Optional[int], Optional[str]], pd.DataFrame
        ] = run_pipeline,
        train_years: YearPair = (None, 2015),
        test_years: YearPair = (2016, 2016),
        start_date: str = "1897-01-01",
        end_date: str = END_OF_YEAR,
    ) -> None:
        super().__init__(
            train_years=train_years,
            test_years=test_years,
            start_date=start_date,
            end_date=end_date,
        )

        self.pipeline_runner = pipeline_runner
        self._data = None

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._data = run_pipeline(self.start_date, self.end_date)

        return self._data
