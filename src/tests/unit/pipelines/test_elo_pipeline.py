from unittest import TestCase

from kedro.pipeline import Pipeline

from machine_learning.pipelines.elo_pipeline import create_elo_pipeline
from machine_learning.pipelines.match_pipeline import create_match_pipeline


class TestEloPipeline(TestCase):
    def test_create_elo_pipeline(self):
        pipeline = create_elo_pipeline(
            create_match_pipeline("2000-01-01", "2010-12-31")
        )

        self.assertIsInstance(pipeline, Pipeline)
        self.assertTrue(any(pipeline.nodes))
