"""Application entry point."""

from typing import Optional
from datetime import date

from kedro.framework.session import KedroSession

from augury import settings


def run_package(
    round_number: Optional[int] = None,
    start_date: str = "1897-01-01",
    end_date: str = f"{date.today().year}-12-31",
):
    """Entry point for running a Kedro project packaged with `kedro package`.

    Uses `python -m <project_package>.run` command.

    Params
    ------
    round_number: The relevant round_number for filtering data.
    start_date: The earliest match date (inclusive) to include in any data sets.
    end_date: The latest match date (inclusive) to include in any data sets.
    """
    extra_params = {
        "round_number": round_number,
        "start_date": start_date,
        "end_date": end_date,
    }
    with KedroSession.create(
        settings.PACKAGE_NAME,
        env=settings.ENV,
        project_path=settings.BASE_DIR,
        extra_params=extra_params,
    ) as session:
        session.run()


if __name__ == "__main__":
    run_package()
