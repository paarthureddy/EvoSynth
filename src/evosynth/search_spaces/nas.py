"""
CNN architecture search space (channels, units, dropout, learning rate, optimizer,
activation). Values come from [search_space.nas] in configs/evosynth.toml.
"""
import ConfigSpace as CS

from evosynth.config import get_settings
from evosynth.search_spaces._build import categorical, numeric


def get_image_nas_search_space(settings=None):
    n = (settings or get_settings()).search_space.nas
    cs = CS.ConfigurationSpace()
    cs.add_hyperparameters([
        categorical("conv1_channels", n.conv1_channels),
        categorical("conv2_channels", n.conv2_channels),
        categorical("fc_units", n.fc_units),
        numeric("dropout_rate", n.dropout_rate),
        numeric("learning_rate", n.learning_rate),
        categorical("optimizer", n.optimizers),
        categorical("activation", n.activations),
    ])
    return cs
