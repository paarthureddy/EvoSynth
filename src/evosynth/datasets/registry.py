"""
Name -> dataset specification + data.

The specification (kind, source, metric, split sizes, ...) lives in [datasets.<name>] of
settings/datasets.toml; the code that fetches the data lives in loaders.py (one loader per
source). `load(name)` keeps the loaded arrays in memory for the rest of the process, and
`partition(name)` the test / search / train split, so the search evaluator, the refit and
every baseline share one copy and exactly the same rows.
"""
from evosynth.config import get_settings
from evosynth.datasets import loaders
from evosynth.datasets.splits import partition as _partition

_CACHE = {}
_PARTS = {}


def available(kind=None, settings=None):
    """Names of configured datasets whose source has a loader (optionally of one kind)."""
    ds = (settings or get_settings()).datasets
    return sorted(n for n, spec in ds.items()
                  if spec.get("source") in loaders.LOADERS and (kind is None or spec.get("kind") == kind))


def get_spec(name, settings=None, kind=None):
    """The [datasets.<name>] settings table (checked to be of `kind` when given)."""
    ds = (settings or get_settings()).datasets
    if name not in ds or ds[name].get("source") not in loaders.LOADERS:
        raise KeyError(f"unknown dataset '{name}'. Available: {available(settings=settings)}")
    if kind is not None and ds[name].get("kind") != kind:
        raise KeyError(f"dataset '{name}' is {ds[name].get('kind')!r}, not {kind!r}. "
                       f"Available {kind} datasets: {available(kind, settings)}")
    return ds[name]


def load(name, settings=None):
    """The whole dataset (see loaders.py for the format). Loaded once per process."""
    if name not in _CACHE:
        spec = get_spec(name, settings)
        _CACHE[name] = loaders.LOADERS[spec.source](name, spec)
    return _CACHE[name]


def partition(name, settings=None):
    """{"search", "train", "test"} → (X, y) for a tabular dataset (splits.partition)."""
    s = settings or get_settings()
    spec = get_spec(name, s, kind="tabular")
    key = (name, s.run.data_seed, spec.test_rows, spec.search_subsample, spec.refit_train_rows)
    if key not in _PARTS:
        X, y = load(name, s)
        need = spec.test_rows + max(spec.search_subsample, spec.refit_train_rows)
        if need > len(y):
            raise ValueError(
                f"dataset '{name}' has {len(y):,} rows, but test_rows ({spec.test_rows:,}) + "
                f"max(search_subsample, refit_train_rows) ({need - spec.test_rows:,}) = {need:,}. "
                f"Lower them in [datasets.{name}] (settings/datasets.toml).")
        _PARTS[key] = _partition(X, y, spec.test_rows, spec.search_subsample, spec.refit_train_rows,
                                 s.run.data_seed)
    return _PARTS[key]
