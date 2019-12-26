from kedro.pipeline import Pipeline, node

from augury.nodes import common, betting, feature_calculation


def create_betting_pipeline(start_date: str, end_date: str, **_kwargs):
    """Kedro pipeline for loading and transforming betting data"""

    return Pipeline(
        [
            node(
                common.convert_to_data_frame,
                ["betting_data", "remote_betting_data"],
                ["betting_data_frame", "remote_betting_data_frame"],
            ),
            node(
                common.combine_data(axis=0),
                ["betting_data_frame", "remote_betting_data_frame"],
                "combined_betting_data",
            ),
            node(betting.clean_data, "combined_betting_data", "clean_betting_data"),
            node(
                common.filter_by_date(start_date, end_date),
                "clean_betting_data",
                "filtered_betting_data",
            ),
            node(
                common.convert_match_rows_to_teammatch_rows,
                "filtered_betting_data",
                "stacked_betting_data",
            ),
            node(
                betting.add_betting_pred_win, ["stacked_betting_data"], "betting_data_a"
            ),
            node(
                feature_calculation.feature_calculator(
                    [
                        (
                            feature_calculation.calculate_rolling_rate,
                            [("betting_pred_win",)],
                        )
                    ]
                ),
                "betting_data_a",
                "betting_data_b",
            ),
            node(
                common.add_oppo_features(
                    oppo_feature_cols=[
                        "betting_pred_win",
                        "rolling_betting_pred_win_rate",
                    ]
                ),
                "betting_data_b",
                "betting_data_c",
            ),
            node(
                common.finalize_data,
                "betting_data_c",
                "prefinal_betting_data",
                name="prefinal_betting_data",
            ),
            node(common.convert_to_json, "prefinal_betting_data", "final_betting_data"),
        ]
    )
