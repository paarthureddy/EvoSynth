"""
The one way every experiment splits a tabular dataset (Phase 3), so all methods see exactly
the same rows and the test rows are never seen by any search:

  1. test   = a stratified sample of test_rows, split off FIRST
  2. search = a stratified sample of search_subsample rows from the rest (search CV)
  3. train  = a stratified sample of refit_train_rows rows from the rest (refit / baselines)

search and train may overlap (both are training data); neither overlaps test.
All three use run.data_seed.
"""
import numpy as np
from sklearn.model_selection import train_test_split


def _stratify(y):
    """Stratify on y unless a class has fewer than 2 rows (then a plain random split)."""
    return y if np.unique(y, return_counts=True)[1].min() >= 2 else None


def sample(X, y, n_rows, random_state):
    """Stratified sample of n_rows rows (all rows, shuffled, if n_rows >= len(y))."""
    if n_rows >= len(y):
        idx = np.random.default_rng(random_state).permutation(len(y))
        return X[idx], y[idx]
    X_s, _, y_s, _ = train_test_split(X, y, train_size=n_rows, stratify=_stratify(y), random_state=random_state)
    return X_s, y_s


def partition(X, y, test_rows, search_rows, train_rows, random_state):
    """{"search": (X, y), "train": (X, y), "test": (X, y)} as described above."""
    X_rest, X_te, y_rest, y_te = train_test_split(X, y, test_size=test_rows, stratify=_stratify(y),
                                                  random_state=random_state)
    return {
        "search": sample(X_rest, y_rest, search_rows, random_state),
        "train": sample(X_rest, y_rest, train_rows, random_state + 1),
        "test": (X_te, y_te),
    }
