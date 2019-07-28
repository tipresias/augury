"""Pipeline construction."""

from kedro.pipeline import Pipeline, node

from machine_learning.data_processors.feature_calculation import (
    feature_calculator,
    calculate_rolling_rate,
)
from machine_learning.data_processors import feature_functions
from .nodes import betting, common


def betting_pipeline(**_kwargs):
    """Kedro pipeline for loading and transforming betting data"""

    return Pipeline(
        [
            node(
                common.convert_to_data_frame,
                ["betting_data", "remote_betting_data"],
                ["betting_data_frame", "remote_betting_data_frame"],
            ),
            node(
                betting.combine_data,
                ["betting_data_frame", "remote_betting_data_frame"],
                "combined_betting_data",
            ),
            node(betting.clean_data, "combined_betting_data", "clean_betting_data"),
            node(
                betting.convert_match_rows_to_teammatch_rows,
                ["clean_betting_data"],
                "stacked_betting_data",
            ),
            node(
                betting.add_betting_pred_win, ["stacked_betting_data"], "betting_data_a"
            ),
            node(
                feature_calculator([(calculate_rolling_rate, [("betting_pred_win",)])]),
                ["betting_data_a"],
                "betting_data_b",
            ),
            node(
                feature_functions.add_oppo_features(
                    oppo_feature_cols=[
                        "betting_pred_win",
                        "rolling_betting_pred_win_rate",
                    ]
                ),
                ["betting_data_b"],
                "betting_data_c",
            ),
            node(betting.finalize_data, ["betting_data_c"], "data"),
        ]
    )


def match_pipeline(**_kwargs):
    """Kedro pipeline for loading and transforming match data"""

    return Pipeline(
        [node(common.convert_to_data_frame, ["match_data"], ["match_data_frame"])]
    )
