"""
Dataset registry. Datasets are downloaded once into .data/ (git-ignored) and read
locally afterwards. Each dataset is described by a [datasets.<name>] section in
settings/datasets.toml and loaded by the loader for its `source` in loaders.py.
"""
from evosynth.datasets.registry import available, get_spec, load, partition
from evosynth.datasets.splits import sample

__all__ = ["available", "get_spec", "load", "partition", "sample"]
