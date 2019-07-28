from unittest import TestCase


import pandas as pd

from tests.fixtures.data_factories import fake_raw_match_results_data
from machine_learning.nodes import common

START_DATE = "2012-01-01"
START_YEAR = int(START_DATE[:4])
END_DATE = "2013-12-31"
END_YEAR = int(END_DATE[:4]) + 1
N_MATCHES_PER_YEAR = 2


class TestCommon(TestCase):
    def test_convert_to_data_frame(self):
        data = fake_raw_match_results_data(
            N_MATCHES_PER_YEAR, (START_YEAR, END_YEAR)
        ).to_dict("records")

        data_frames = common.convert_to_data_frame(data, data)

        self.assertEqual(len(data_frames), 2)

        for data_frame in data_frames:
            self.assertIsInstance(data_frame, pd.DataFrame)

        raw_data_fields = data[0].keys()
        data_frame_columns = data_frames[0].columns

        self.assertEqual(set(raw_data_fields), set(data_frame_columns))
