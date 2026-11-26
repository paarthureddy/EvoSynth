"""
One seeding scheme for the whole project (Phase 0).

Two settings in settings/run.toml:
  run.seed       search randomness: optimizers, initial population, injections, model
                 random_state, Optuna / FLAML seeds, PyTorch. Varies between repeated runs.
  run.data_seed  data randomness: search subsample, CV folds, refit train/test split.
                 Kept fixed so every seed of every method is scored on the same data.

Every random component draws from its OWN numpy Generator, derived from run.seed plus a
name (stream("optimizer", "PSO", 3)). Streams are independent of each other, so a result
never depends on the order in which components happen to draw numbers, and adding a draw
in one component does not shift another's. Global generators (numpy, random, torch) are
also seeded at the start of each run, as a safety net for any library that uses them.
"""
import random
import zlib

import numpy as np

from evosynth.config import get_settings


def run_seed(settings=None):
    return (settings or get_settings()).run.seed


def data_seed(settings=None):
    return (settings or get_settings()).run.data_seed


def _key(part):
    return part if isinstance(part, int) and part >= 0 else zlib.crc32(str(part).encode("utf-8"))


def stream(*names, seed=None):
    """Independent numpy Generator for the component identified by `names`."""
    seed = run_seed() if seed is None else seed
    return np.random.default_rng(np.random.SeedSequence([seed, *(_key(n) for n in names)]))


def derive_int(*names, seed=None, high=2**31 - 1):
    """A reproducible integer seed (for libraries that take an int, e.g. torch, cma)."""
    return int(stream(*names, seed=seed).integers(1, high))


def seed_everything(seed=None):
    """Seed the global generators (numpy, random, torch if installed). Called by RunContext."""
    seed = run_seed() if seed is None else seed
    random.seed(seed)
    np.random.seed(seed % 2**32)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    return seed
