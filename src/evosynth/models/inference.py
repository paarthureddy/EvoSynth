"""
Inference cost of a fitted model (Phase 4.8): measured, not claimed.

  single_row_ms   median wall time of predict() on one row, on one thread (repeated [inference].single_row_repeats
                  times on different test rows; first call excluded as warm-up)
  batch_rows_per_s  rows per second when predicting the whole test set at once
  model_mb        size of the pickled fitted model (feature engineering not included)

Settings: [inference] (settings/evaluation.toml).
"""
import pickle
import time

import numpy as np

from evosynth.config import get_settings


def _threads(model, n_jobs):
    """Set n_jobs on the (possibly wrapped) estimator; returns the previous value or None."""
    inner = getattr(model, "model", model)
    if not hasattr(inner, "get_params") or "n_jobs" not in inner.get_params():
        return None
    old = inner.get_params()["n_jobs"]
    inner.set_params(n_jobs=n_jobs)
    return old


def measure(model, X_test, settings=None):
    s = (settings or get_settings()).inference
    n = min(s.single_row_repeats, len(X_test))
    # A single-row request is served on one thread: with n_jobs=-1, scikit-learn's forests spend
    # ~15 ms starting threads for one row, which is not the model's cost.
    old = _threads(model, 1)
    model.predict(X_test[:1])                                   # warm-up
    times = []
    for i in range(n):
        t = time.perf_counter()
        model.predict(X_test[i:i + 1])
        times.append(time.perf_counter() - t)
    if old is not None:
        _threads(model, old)
    t = time.perf_counter()
    model.predict(X_test)
    batch = time.perf_counter() - t
    return {
        "single_row_ms": round(float(np.median(times)) * 1000.0, 3),
        "single_row_p95_ms": round(float(np.percentile(times, 95)) * 1000.0, 3),
        "batch_rows": int(len(X_test)),
        "batch_rows_per_s": round(len(X_test) / max(batch, 1e-9), 1),
        "model_mb": round(len(pickle.dumps(model)) / 1e6, 3),
    }
