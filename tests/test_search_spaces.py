"""Search spaces built from settings and decoding of latent vectors."""
import numpy as np
import pytest

from evosynth.core.decoding import decode_vector_to_dict
from evosynth.search_spaces.nas import get_image_nas_search_space
from evosynth.search_spaces.tabular import get_tabular_search_space
from evosynth.search_spaces.toy import get_dummy_search_space

TABULAR_KEYS = ["booster", "colsample_bytree", "fe_strategy", "learning_rate", "max_depth",
                "model_type", "n_estimators", "reg_alpha", "subsample"]


def test_tabular_space_dimensions_are_alphabetical():
    assert list(get_tabular_search_space().keys()) == TABULAR_KEYS
    assert len(get_tabular_search_space(include_model_type=False, include_fe_strategy=False).keys()) == 7


def test_other_spaces():
    assert len(get_image_nas_search_space().keys()) == 7
    assert len(get_dummy_search_space().keys()) == 4


@pytest.mark.parametrize("z, expected", [(0.0, "XGBoost"), (0.30, "RandomForest"),
                                         (0.60, "LightGBM"), (0.99, "ExtraTrees"), (1.0, "ExtraTrees")])
def test_categorical_binning(z, expected):
    space = get_tabular_search_space()
    v = np.full(len(TABULAR_KEYS), 0.5)
    v[TABULAR_KEYS.index("model_type")] = z
    assert decode_vector_to_dict(v, space)["model_type"] == expected


def test_decoded_values_stay_in_range(default_settings):
    space = get_tabular_search_space()
    rng = np.random.default_rng(0)
    t = default_settings.search_space.tabular
    for _ in range(50):
        cfg = decode_vector_to_dict(rng.random(len(TABULAR_KEYS)), space)
        assert t.learning_rate.low <= cfg["learning_rate"] <= t.learning_rate.high
        assert t.n_estimators.low <= cfg["n_estimators"] <= t.n_estimators.high
        assert cfg["fe_strategy"] in t.fe_strategies


def test_range_comes_from_settings():
    from evosynth.config import load_config, set_settings
    set_settings(load_config(overrides=["search_space.tabular.max_depth={low=2, high=4, int=true}"]))
    space = get_tabular_search_space()
    v = np.ones(len(TABULAR_KEYS))
    assert decode_vector_to_dict(v, space)["max_depth"] == 4
