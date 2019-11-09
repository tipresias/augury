from typing import Dict

from kedro.pipeline import Pipeline

from tests.fixtures.fake_pipeline import create_fake_pipeline
from machine_learning.nodes import feature_calculation
from .player_pipeline import create_player_pipeline
from .betting_pipeline import create_betting_pipeline
from .match_pipeline import create_match_pipeline
from .full_pipeline import create_full_pipeline


def create_pipelines(start_date, end_date, **_kwargs) -> Dict[str, Pipeline]:
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """

    return {
        "__default__": Pipeline([]),
        "betting": create_betting_pipeline(start_date, end_date),
        "match": create_match_pipeline(start_date, end_date),
        "player": create_player_pipeline(start_date, end_date),
        "full": create_full_pipeline(start_date, end_date),
        "fake": create_fake_pipeline(),
    }
