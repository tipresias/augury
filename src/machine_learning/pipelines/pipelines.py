from typing import Dict

from kedro.pipeline import Pipeline

from tests.fixtures.fake_pipeline import create_fake_pipeline
from machine_learning.nodes import feature_calculation
from .player_pipeline import create_player_pipeline
from .betting_pipeline import create_betting_pipeline
from .match_pipeline import create_match_pipeline, create_legacy_match_pipeline
from .full_pipeline import create_full_pipeline
from .elo_pipeline import create_elo_pipeline


LEGACY_FEATURE_CALCS = [
    (feature_calculation.calculate_division, [("elo_rating", "win_odds")]),
    (feature_calculation.calculate_multiplication, [("win_odds", "ladder_position")]),
]


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
        "legacy": create_full_pipeline(
            start_date,
            end_date,
            match_pipeline_func=create_legacy_match_pipeline,
            feature_calcs=LEGACY_FEATURE_CALCS,
            final_data_set="legacy_model_data",
        ),
        "elo": create_elo_pipeline(
            match_pipeline=create_match_pipeline(start_date, end_date),
        ),
        "fake": create_fake_pipeline(),
    }
