from unittest import TestCase
import pandas as pd
from machine_learning.ml_data import MLData


class TestMLData(TestCase):
    def setUp(self):
        self.ml_data = MLData

    # full includes betting, match, and player, so no reason to test them separately
    def test_full_pipeline(self):
        ml_data = self.ml_data(
            pipeline="full",
            # We don't use any data set, but this makes sure we don't overwrite
            # one that actually matters
            data_set="fake_data",
            train_years=(2014, 2014),
            test_years=(2015, 2015),
            start_date="2014-01-01",
            end_date="2015-12-31",
            update_data=True,
            # This stops just short of the step that writes to a JSON file
            to_nodes=["final_model_data"],
        )

        self.assertIsInstance(ml_data.data, pd.DataFrame)

    def test_legacy_pipeline(self):
        ml_data = self.ml_data(
            pipeline="legacy",
            # We don't use any data set, but this makes sure we don't overwrite
            # one that actually matters
            data_set="fake_data",
            train_years=(2014, 2014),
            test_years=(2015, 2015),
            start_date="2014-01-01",
            end_date="2015-12-31",
            update_data=True,
            # This stops just short of the step that writes to a JSON file
            to_nodes=["final_model_data"],
        )

        self.assertIsInstance(ml_data.data, pd.DataFrame)
