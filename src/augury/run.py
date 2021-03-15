"""Application entry point."""

from pathlib import Path
from typing import Optional
from datetime import date

from kedro.framework.session import KedroSession

from augury.settings import BASE_DIR


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
    # Entry point for running a Kedro project packaged with `kedro package`
    # using `python -m <project_package>.run` command.
    package_name = Path(__file__).resolve().parent.name
    extra_params = {
        "round_number": round_number,
        "start_date": start_date,
        "end_date": end_date,
    }
    with KedroSession.create(
        package_name, project_path=BASE_DIR, extra_params=extra_params
    ) as session:
        session.run()


if __name__ == "__main__":
    run_package()
