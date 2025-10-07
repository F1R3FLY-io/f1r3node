import os
from pathlib import Path
from typing import Iterable

import yaml

GITHUB_DIR = os.path.dirname(os.path.abspath(__file__))
REQUIRED_WORKFLOW_PATH = GITHUB_DIR + '/workflows/build-test-and-deploy.yml'
OPTIONAL_WORKFLOW_PATH = GITHUB_DIR + '/workflows/optional-tests.yml'
REMAINDER = 'REMAINDER'


def get_project_root() -> Path:
    """Returns project root path from PROJECT_ROOT environment variable or
    falls back to current working directory"""
    return Path(os.getenv('PROJECT_ROOT', os.getcwd()))


def read_workflow(workflow_path: Path) -> dict:
    """Reads YAML workflow as dict"""
    with open(workflow_path) as f:
        return yaml.safe_load(f)


def get_test_selections() -> Iterable[str]:
    project_root = get_project_root()
    required_workflow_path = project_root / REQUIRED_WORKFLOW_PATH
    optional_workflow_path = project_root / OPTIONAL_WORKFLOW_PATH
    required_workflow = read_workflow(required_workflow_path)
    optional_workflow = read_workflow(optional_workflow_path)
    selections = \
        required_workflow['jobs']['run_required_scala_unit_tests']['strategy']['matrix']['tests'] + \
        optional_workflow['jobs']['run_optional_scala_unit_tests']['strategy']['matrix']['tests']

    remainder_found = False

    for test_sel in selections:
        if test_sel == REMAINDER:
            if not remainder_found:
                remainder_found = True
                continue
            raise Exception(f"Two {REMAINDER} test selections found")
        yield test_sel
