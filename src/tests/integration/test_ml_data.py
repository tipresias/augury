from unittest import TestCase, skipIf
import os
import pandas as pd
from augury.ml_data import MLData


class TestMLData(TestCase):
    def setUp(self):
        self.ml_data = MLData

    @skipIf(
        os.getenv("CI", "").lower() == "true",
        "More trouble than it's worth trying to get usable data sets in CI."
        "Also, given my difficulties getting player data loaded in CI, running this "
        "is likely equally impossible.",
    )
    # full includes betting, match, and player, so no reason to test them separately
    def test_full_pipeline(self):
        ml_data = self.ml_data(
            pipeline="full",
            # We don't use any data set, but this makes sure we don't overwrite
            # one that actually matters
            data_set="fake_data",
            train_year_range=(2014, 2015),
            test_year_range=(2015, 2016),
            start_date="2014-01-01",
            end_date="2015-12-31",
            update_data=True,
            # This stops just short of the step that writes to a JSON file
            to_nodes=["final_model_data"],
        )

        self.assertIsInstance(ml_data.data, pd.DataFrame)

    @skipIf(
        os.getenv("CI", "").lower() == "true",
        "More trouble than it's worth trying to get usable data sets in CI."
        "Also, given my difficulties getting player data loaded in CI, running this "
        "is likely equally impossible.",
    )
    def test_legacy_pipeline(self):
        ml_data = self.ml_data(
            pipeline="legacy",
            # We don't use any data set, but this makes sure we don't overwrite
            # one that actually matters
            data_set="fake_data",
            train_year_range=(2014, 2015),
            test_year_range=(2015, 2016),
            start_date="2014-01-01",
            end_date="2015-12-31",
            update_data=True,
            # This stops just short of the step that writes to a JSON file
            to_nodes=["final_model_data"],
        )

        self.assertIsInstance(ml_data.data, pd.DataFrame)
