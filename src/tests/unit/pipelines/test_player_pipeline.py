from unittest import TestCase

from kedro.pipeline import Pipeline

from augury.pipelines.player_pipeline import create_player_pipeline


class TestPlayerPipeline(TestCase):
    def test_create_player_pipeline(self):
        pipeline = create_player_pipeline("2000-01-01", "2010-12-31")

        self.assertIsInstance(pipeline, Pipeline)
        self.assertTrue(any(pipeline.nodes))
