import json
from pathlib import Path

import pytest

from dermis.telemetry.sources import find_fixture


def _fixture(name):
    p = find_fixture(name)
    if p is None:
        pytest.skip(f"supplied fixture {name} not found in repo")
    return p


@pytest.fixture(scope="session")
def decoded_frame():
    return json.loads(_fixture("decoded_frame.json").read_text())


@pytest.fixture(scope="session")
def fixture_path():
    return _fixture
