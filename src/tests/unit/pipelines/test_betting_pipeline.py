from unittest import TestCase

from kedro.pipeline import Pipeline

from augury.pipelines.betting_pipeline import create_betting_pipeline


class TestBettingPipeline(TestCase):
    def test_create_betting_pipeline(self):
        pipeline = create_betting_pipeline("2000-01-01", "2010-12-31")

        self.assertIsInstance(pipeline, Pipeline)
        self.assertTrue(any(pipeline.nodes))
