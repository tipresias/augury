"""Pipeline construction."""

import os

from kedro.pipeline import Pipeline, node

from machine_learning.settings import CATEGORY_COLS, BASE_DIR
from machine_learning.nodes import common, feature_calculation
from .player_pipeline import create_player_pipeline
from .betting_pipeline import create_betting_pipeline
from .match_pipeline import create_match_pipeline


DEFAULT_FEATURE_CALCS = [
    (feature_calculation.calculate_multiplication, [("win_odds", "ladder_position")],),
]


def create_full_pipeline(
    start_date: str,
    end_date: str,
    match_pipeline_func=create_match_pipeline,
    feature_calcs=DEFAULT_FEATURE_CALCS,
):
    return Pipeline(
        [
            create_betting_pipeline(start_date, end_date),
            match_pipeline_func(start_date, end_date),
            create_player_pipeline(start_date, end_date),
            node(
                common.combine_data(axis=1),
                ["final_betting_data", "final_match_data", "final_player_data"],
                "joined_data",
            ),
            node(
                feature_calculation.feature_calculator(feature_calcs),
                "joined_data",
                "data_a",
            ),
            node(
                common.sort_data_frame_columns(category_cols=CATEGORY_COLS),
                "data_a",
                "data_b",
            ),
            node(common.finalize_data, "data_b", "data_c"),
            node(
                common.convert_to_json(
                    os.path.join(
                        BASE_DIR,
                        "data/05_model_input/",
                        f"model-data_{start_date}_{end_date}.json",
                    )
                ),
                "data_c",
                "model_data",
            ),
        ]
    )
