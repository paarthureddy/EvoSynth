"""Shared fixtures: every test starts from the default settings in settings/."""
import sys

import pytest

from evosynth.config import load_config, set_settings


@pytest.fixture(autouse=True)
def default_settings():
    cfg = load_config()
    set_settings(cfg)
    stdout = sys.stdout
    yield cfg
    set_settings(load_config())
    sys.stdout = stdout          # a failed run must never leave stdout redirected


@pytest.fixture
def toy_problem():
    """Fast synthetic objective (microseconds per evaluation) and its 4-D space."""
    from evosynth.evaluators.toy import DummyObjectiveFunction
    evaluator = DummyObjectiveFunction()
    return evaluator.space, evaluator
