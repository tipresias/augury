import os
import sys
from datetime import date

import pandas as pd
import numpy as np

BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from machine_learning import api
from machine_learning.ml_data import MLData
from machine_learning.data_import import match_data
from machine_learning.data_config import INDEX_COLS, SEED

np.random.seed(SEED)

FIRST_YEAR_WITH_CURRENT_TEAMS = 2012


def _predicted_home_margins(df, pred_col="predicted_margin"):
    home_teams = (
        df.query("at_home == 1").rename_axis(INDEX_COLS + ["ml_model"]).sort_index()
    )
    away_teams = (
        df.query("at_home == 0 & team != 0")
        .reset_index(drop=False)
        .set_index(["oppo_team", "year", "round_number", "ml_model"])
        .rename_axis(INDEX_COLS + ["ml_model"])
        .sort_index()
    )

    home_margin_multiplier = (home_teams[pred_col] > away_teams[pred_col]).map(
        lambda x: 1 if x else -1
    )

    return (
        pd.Series(
            ((home_teams[pred_col].abs() + away_teams[pred_col].abs()) / 2)
            * home_margin_multiplier
        )
        .reindex(home_teams.index)
        .sort_index()
    )


def _calculate_correct(y, y_pred):
    # Could give half point for predicted a draw (y_pred == 0), but it only happens
    # with betting odds, and only rarely, so it's easier to just give it to them
    return (y == 0) | ((y >= 0) & (y_pred >= 0)) | ((y <= 0) & (y_pred <= 0))


def _betting_predictions(data):
    # We use oppo_line_odds for predicted_margin, because in betting odds
    # low is favoured, high is underdog
    return (
        data.data.query("year >= @FIRST_YEAR_WITH_CURRENT_TEAMS")[
            [
                "team",
                "year",
                "round_number",
                "oppo_team",
                "at_home",
                "oppo_line_odds",
                "margin",
            ]
        ]
        .assign(ml_model="betting_odds")
        .rename(columns={"oppo_line_odds": "predicted_margin"})
        .set_index(INDEX_COLS + ["ml_model"])
    )


def _model_predictions(data, year_range):
    return pd.DataFrame(
        api.make_predictions(year_range, data=data, train=True).get("data")
    ).set_index(INDEX_COLS + ["ml_model"])


def _predictions(data, year_range):
    predictions = pd.concat(
        [_model_predictions(data, year_range), _betting_predictions(data)], sort=False
    ).query("team != 0")

    model_names = predictions.index.get_level_values(level="ml_model").drop_duplicates()
    model_margins = [
        data.data[["margin"]].assign(ml_model=model_name) for model_name in model_names
    ]

    match_margins = (
        pd.concat(model_margins, sort=False)
        .rename_axis(INDEX_COLS)
        .reset_index(drop=False)
        .set_index(INDEX_COLS + ["ml_model"])
    )

    return predictions.assign(margin=match_margins["margin"])


def main():
    year_range = (FIRST_YEAR_WITH_CURRENT_TEAMS, date.today().year + 1)
    year_range_label = f"{year_range[0]}_{year_range[1] - 1}"

    data = MLData(
        start_date=f"{match_data.FIRST_YEAR_OF_MATCH_DATA}-01-01",
        end_date=f"{api.END_OF_YEAR}",
    )
    predictions = _predictions(data, year_range)

    home_team_predictions = _predicted_home_margins(predictions)
    home_team_margins = pd.concat(
        [home_team_predictions, predictions["margin"]], join="inner", sort=False, axis=1
    ).loc[:, "margin"]
    correct_predictions = _calculate_correct(home_team_margins, home_team_predictions)

    predictions.query("at_home == 1").assign(
        predicted_home_win=lambda df: df["predicted_margin"] >= 0,
        is_correct=correct_predictions,
    ).reset_index(drop=False).rename(
        columns={
            "team": "home_team",
            "oppo_team": "away_team",
            "predicted_margin": "predicted_home_margin",
        }
    ).loc[
        :,
        [
            "home_team",
            "away_team",
            "year",
            "round_number",
            "ml_model",
            "predicted_home_margin",
            "predicted_home_win",
            "is_correct",
        ],
    ].to_json(
        f"{BASE_DIR}/data/07_model_output/model_predictions_{year_range_label}.json",
        orient="records",
    )


if __name__ == "__main__":
    main()
