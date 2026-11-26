"""settings/ folder: loading, merging, presets, overrides and validation."""
import shutil

import pytest

from evosynth.config import ConfigError, get_settings, load_config, read_settings
from evosynth.paths import SETTINGS_DIR

SECTIONS = {"bandit", "composite", "datasets", "engine", "evaluation", "experiments", "features",
            "optimizers", "refit", "run", "safeguards", "search_space", "telemetry"}


def test_folder_loads_all_sections():
    assert set(read_settings()) == SECTIONS


def test_defaults_are_valid_and_sensible():
    s = load_config()
    w = s.composite
    assert abs(w.fitness + w.speed + w.efficiency + w.diversity + w.trend - 1.0) < 1e-9
    assert s.engine.n_promoted == 2
    assert s.bandit.c == pytest.approx(0.1)
    assert "covertype" in s.datasets


def test_preset_overrides_engine():
    s = load_config(preset="search_demo")
    assert (s.engine.budget, s.engine.pop_size, s.engine.t1_iters, s.engine.k_iters) == (200, 40, 3, 2)


def test_dotted_override_is_parsed_as_toml():
    s = load_config(overrides=["bandit.c=0.25", "experiments.scaling.budgets=[40, 80]"])
    assert s.bandit.c == pytest.approx(0.25)
    assert list(s.experiments.scaling.budgets) == [40, 80]


@pytest.mark.parametrize("override", ["bandit.nope=1", "nosection.key=1", "bandit.c"])
def test_bad_override_is_rejected(override):
    with pytest.raises(ConfigError):
        load_config(overrides=[override])


@pytest.mark.parametrize("override", [
    "composite.fitness=0.9",            # weights no longer sum to 1
    "engine.pop_size=20",               # Tier 1 islands would have fewer than 4 members
    "experiments.nas.pop_size=20",      # same check for every preset
    "engine.tier_split=\"best\"",       # unknown split method
    "engine.budget=10",                 # budget must exceed the population
])
def test_invalid_values_are_rejected(override):
    with pytest.raises(ConfigError):
        load_config(overrides=[override])


def test_duplicate_section_across_files_is_an_error(tmp_path):
    for f in SETTINGS_DIR.glob("*.toml"):
        shutil.copy(f, tmp_path / f.name)
    (tmp_path / "extra.toml").write_text("[bandit]\nc = 0.3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="bandit"):
        read_settings(tmp_path)


def test_get_settings_returns_active(default_settings):
    assert get_settings().engine.budget == default_settings.engine.budget
