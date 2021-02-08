"""Application entry point."""

from pathlib import Path
from typing import Optional
from datetime import date
import os

from kedro.framework.context import KedroContext, load_package_context

from augury.settings import BASE_DIR


class ProjectContext(KedroContext):
    """Specialisation of generic KedroContext object with params specific to Augury."""

    project_name = "augury"
    project_version = "0.16.5"

    def __init__(
        self,
        project_path: str,
        env: Optional[str] = os.getenv("PYTHON_ENV"),
        round_number: Optional[int] = None,
        start_date: str = "1897-01-01",
        end_date: str = f"{date.today().year}-12-31",
        **kwargs,
    ):
        """
        Instantiate ProjectContext object.

        Params
        ------
        project_path: Absolute path to project root.
        env: Name of the current environment. Principally used
            to load the correct `conf/` files.
        round_number: The relevant round_number for filtering data.
        start_date: The earliest match date (inclusive) to include in any data sets.
        end_date: The latest match date (inclusive) to include in any data sets.
        """
        super().__init__(project_path, env=env, **kwargs)
        self.round_number = round_number
        self.start_date = start_date
        self.end_date = end_date


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
    project_context = load_package_context(
        project_path=Path(BASE_DIR),
        package_name=Path(__file__).resolve().parent.name,
        round_number=round_number,
        start_date=start_date,
        end_date=end_date,
    )
    project_context.run()


if __name__ == "__main__":
    run_package()
