"""
Name -> dataset specification + loader.

The specification (source, metric, subsample sizes, ...) lives in
[datasets.<name>] of settings/datasets.toml; the code that fetches the data lives in
loaders.py. `load(name)` keeps the loaded arrays in memory for the rest of the process,
so the search evaluator and the post-search refit share one copy.
"""
from evosynth.config import get_settings
from evosynth.datasets import loaders

# dataset name -> loader function (receives the dataset's settings table)
_LOADERS = {
    "covertype": loaders.load_covertype,
}

_CACHE = {}


def available():
    """Names of datasets that have both a loader and a [datasets.<name>] section."""
    configured = set(get_settings().datasets)
    return sorted(n for n in _LOADERS if n in configured)


def get_spec(name, settings=None):
    """The [datasets.<name>] settings table."""
    ds = (settings or get_settings()).datasets
    if name not in ds or name not in _LOADERS:
        raise KeyError(f"unknown dataset '{name}'. Available: {available()}")
    return ds[name]


def load(name, settings=None):
    """Full dataset as (X, y). Loaded once per process, then served from memory."""
    if name not in _CACHE:
        spec = get_spec(name, settings)
        _CACHE[name] = _LOADERS[name](spec)
    return _CACHE[name]
