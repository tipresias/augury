from unittest import TestCase

from kedro.pipeline import Pipeline

from augury.pipelines import create_pipelines


class TestPipelines(TestCase):
    def test_create_pipelines(self):
        pipelines_dict = create_pipelines("2000-01-01", "2010-12-31")

        for pipeline in pipelines_dict.values():
            self.assertIsInstance(pipeline, Pipeline)
