"""
4-D toy space for the DummyObjectiveFunction (unit tests only; no dataset).
Values come from [search_space.toy] in settings/search_spaces.toml.
"""
import ConfigSpace as CS

from evosynth.config import get_settings
from evosynth.search_spaces._build import categorical, numeric


def get_dummy_search_space(settings=None):
    t = (settings or get_settings()).search_space.toy
    cs = CS.ConfigurationSpace()
    cs.add_hyperparameters([
        numeric("learning_rate", t.learning_rate),
        numeric("num_layers", t.num_layers),
        categorical("optimizer", t.optimizers),
        numeric("dropout", t.dropout),
    ])
    return cs
