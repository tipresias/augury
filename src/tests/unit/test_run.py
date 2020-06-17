# pylint: disable=missing-module-docstring, missing-function-docstring
# pylint: disable=missing-class-docstring

from pathlib import Path

import pytest

from augury.run import ProjectContext


@pytest.fixture
def project_context():
    return ProjectContext(str(Path.cwd()))


class TestProjectContext:
    @staticmethod
    def test_project_name(project_context):  # pylint: disable=redefined-outer-name
        assert project_context.project_name == "augury"

    @staticmethod
    def test_project_version(project_context):  # pylint: disable=redefined-outer-name
        assert project_context.project_version == "0.16.1"
