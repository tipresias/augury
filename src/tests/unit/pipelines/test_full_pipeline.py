from unittest import TestCase

from kedro.pipeline import Pipeline

from augury.pipelines.full_pipeline import create_full_pipeline


class TestFullPipeline(TestCase):
    def test_create_full_pipeline(self):
        pipeline = create_full_pipeline("2000-01-01", "2010-12-31")

        self.assertIsInstance(pipeline, Pipeline)
        self.assertTrue(any(pipeline.nodes))
