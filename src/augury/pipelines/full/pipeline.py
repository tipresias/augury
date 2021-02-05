"""Pipelines for loading and processing joined data from other pipelines."""

from typing import List

from kedro.pipeline import Pipeline, node

from augury.settings import CATEGORY_COLS
from augury.types import CalculatorPair
from ..nodes import common, feature_calculation
from .. import player, betting, match

FEATURE_CALCS: List[CalculatorPair] = [
    (feature_calculation.calculate_multiplication, [("win_odds", "ladder_position")])
]
PIPELINE_CATEGORY_COLS = CATEGORY_COLS + [
    "prev_match_oppo_team",
    "oppo_prev_match_oppo_team",
]


def create_pipeline(
    start_date: str,
    end_date: str,
):
    """Create a pipeline that joins all data-source-specific pipelines."""
    return Pipeline(
        [
            betting.create_pipeline(start_date, end_date),
            match.create_pipeline(start_date, end_date),
            player.create_pipeline(start_date, end_date),
            node(
                common.clean_full_data,
                ["final_betting_data", "final_match_data", "final_player_data"],
                [
                    "clean_betting_data_df",
                    "clean_match_data_df",
                    "clean_player_data_df",
                ],
            ),
            node(
                common.combine_data(axis=1),
                [
                    "clean_betting_data_df",
                    "clean_match_data_df",
                    "clean_player_data_df",
                ],
                "joined_data",
                name="joined_data",
            ),
            node(
                common.filter_by_date(start_date, end_date),
                "joined_data",
                "filtered_data",
            ),
            node(
                feature_calculation.feature_calculator(FEATURE_CALCS),
                "filtered_data",
                "data_a",
            ),
            node(
                common.sort_data_frame_columns(category_cols=PIPELINE_CATEGORY_COLS),
                "data_a",
                "sorted_data",
            ),
            node(common.finalize_data, "sorted_data", "model_data"),
        ],
    )
