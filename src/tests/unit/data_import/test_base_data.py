from unittest import TestCase

import responses

from machine_learning.data_import.base_data import (
    fetch_afl_data,
    LOCAL_AFL_DATA_SERVICE,
)


FAKE_JSON = [{"number": 5, "name": "bob"}, {"number": 454, "name": "jim"}]


class TestBaseData(TestCase):
    def setUp(self):
        self.add_success_response = lambda: responses.add(
            responses.GET, f"{LOCAL_AFL_DATA_SERVICE}/data", json=FAKE_JSON, status=200
        )
        self.add_failure_response = lambda: responses.add(
            responses.GET,
            f"{LOCAL_AFL_DATA_SERVICE}/data",
            json={"error": "bad things"},
            status=500,
        )

    @responses.activate
    def test_fetch_afl_data(self):
        self.add_success_response()

        res = fetch_afl_data("/data")
        self.assertEqual(res, FAKE_JSON)

        with self.subTest("when first response isn't 200"):
            self.add_failure_response()
            self.add_success_response()

            res = fetch_afl_data("/data")
            self.assertEqual(res, FAKE_JSON)

        with self.subTest("when the retry returns a failure response"):
            self.add_failure_response()
            self.add_failure_response()

            with self.assertRaises(Exception):
                fetch_afl_data("/data")
