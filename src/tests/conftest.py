# pylint: disable=missing-module-docstring,missing-function-docstring
import pytest
from kedro.framework.session import KedroSession

from augury import settings


@pytest.fixture(scope="session", autouse=True)
def kedro_session():
    with KedroSession.create(
        settings.PACKAGE_NAME, project_path=settings.BASE_DIR, env=settings.ENV
    ) as session:
        yield session
