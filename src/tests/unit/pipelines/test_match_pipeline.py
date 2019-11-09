from unittest import TestCase

from kedro.pipeline import Pipeline

from machine_learning.pipelines.match_pipeline import create_match_pipeline


class TestMatchPipeline(TestCase):
    def test_create_match_pipeline(self):
        pipeline = create_match_pipeline("2000-01-01", "2010-12-31")

        self.assertIsInstance(pipeline, Pipeline)
        self.assertTrue(any(pipeline.nodes))
