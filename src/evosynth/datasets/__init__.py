"""
Dataset registry. Datasets are downloaded once into .data/ (git-ignored) and read
locally afterwards. Each dataset is described by a [datasets.<name>] section in
configs/evosynth.toml and a loader function in loaders.py.
"""
from evosynth.datasets.registry import available, get_spec, load
from evosynth.datasets.splits import refit_split, search_subsample

__all__ = ["available", "get_spec", "load", "refit_split", "search_subsample"]
