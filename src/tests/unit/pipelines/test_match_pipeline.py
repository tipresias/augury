# pylint: disable=missing-module-docstring, missing-function-docstring
# pylint: disable=missing-class-docstring

from unittest import TestCase

from kedro.pipeline import Pipeline

from augury.pipelines.match import create_pipeline


class TestMatchPipeline(TestCase):
    def test_create_pipeline(self):
        pipeline = create_pipeline("2000-01-01", "2010-12-31")

        self.assertIsInstance(pipeline, Pipeline)
        self.assertTrue(any(pipeline.nodes))
