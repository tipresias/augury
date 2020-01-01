"""Functions and classes to deduplicate and simplify test code"""

import os

from kedro.context import load_context

from augury.settings import BASE_DIR


class KedroContextMixin:
    """Mixin class for loading the kedro context in tests"""

    @staticmethod
    def load_context(**context_kwargs):
        """
        Load the kedro context, changing the environment variable for CI,
        so data sets will load correctly.
        """

        # Need to use production environment for loading data sets if in CI, because we
        # don't check data set files into source control
        kedro_env = (
            "production"
            if os.environ.get("CI") == "true"
            else os.environ.get("PYTHON_ENV")
        )

        return load_context(BASE_DIR, env=kedro_env, **context_kwargs)
