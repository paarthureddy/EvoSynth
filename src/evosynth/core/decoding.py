"""
Latent-vector decoding.

A latent vector z in [0,1]^d is turned into real hyperparameter values:
categorical dimensions are binned as min(floor(z*K), K-1) (the index ConfigSpace
expects), all other dimensions are passed through as ConfigSpace's normalised
vector values. Dimension order is `list(space.keys())`.
"""
import numpy as np
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import Configuration


def decode_vector_to_dict(vector, space):
    """Decode a latent vector into a {hyperparameter_name: value} dict."""
    valid_vec = np.copy(vector)
    for i, hp_name in enumerate(list(space.keys())):
        hp = space.get_hyperparameter(hp_name)
        if isinstance(hp, CSH.CategoricalHyperparameter):
            num_choices = len(hp.choices)
            idx = int(np.floor(valid_vec[i] * num_choices))
            idx = min(idx, num_choices - 1)
            valid_vec[i] = float(idx)
    cfg = Configuration(space, vector=valid_vec)
    return {k: cfg[k] for k in list(space.keys())}
