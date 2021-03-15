"""Customizations for Kedro's context module."""

from typing import Optional
from datetime import date
import os

from kedro.framework.context import load_context

from augury.settings import BASE_DIR, PREDICTION_DATA_START_DATE


END_OF_YEAR = f"{date.today().year}-12-31"


def load_project_context(round_number: Optional[int] = None, **context_kwargs):
    """Load a Kedro context specific to this project and the current environment."""
    kedro_env = os.environ.get("PYTHON_ENV") or "local"

    date_kwargs = (
        {"start_date": PREDICTION_DATA_START_DATE, "end_date": END_OF_YEAR}
        if os.getenv("PYTHON_ENV", "").lower() == "production"
        else {}
    )

    return load_context(
        BASE_DIR,
        env=kedro_env,
        extra_params={
            "round_number": round_number,
            **date_kwargs,
        },
        **context_kwargs,
    )
