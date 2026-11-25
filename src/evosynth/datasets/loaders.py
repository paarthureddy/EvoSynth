"""
One loader per dataset. Each returns (X, y) as numpy arrays (X float32, y int labels
starting at 0) and caches raw downloads in evosynth.paths.DATA_DIR (.data/).
New datasets: add a function here and register it in registry.py + settings/datasets.toml.
"""
import numpy as np

from evosynth.paths import DATA_DIR


def load_covertype(spec):
    """UCI Forest Covertype (581,012 x 54, 7 classes) via sklearn.datasets.fetch_covtype."""
    from sklearn.datasets import fetch_covtype
    data = fetch_covtype(data_home=str(DATA_DIR))
    X = data.data.astype(np.float32)
    y = data.target.astype(int) - int(spec["label_offset"])
    return X, y
