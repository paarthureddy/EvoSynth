"""
Speed-up benchmark (Phase 4.9). Command: `evosynth speedup [--dataset covertype]`.

Re-measures the "parallel workers + early stopping" speed-up on a real dataset (the v1 claim of
11.5x was measured on a synthetic 50k x 25 dataset). The same `configurations` random
configurations (seeded) are evaluated four times, exactly as the search evaluates a population:

  sequential, no early stopping   1 worker,  boosted models train every round  (reference)
  sequential, early stopping      1 worker,  [evaluation] early stopping
  parallel,   no early stopping   `workers` threads
  parallel,   early stopping      `workers` threads + early stopping  (EvoSynth's default)

Reports wall time, speed-up over the reference, and mean CV score (early stopping changes the
scores slightly, so the speed-up is not free). Settings: [experiments.speedup].

Output: results/speedup/<dataset>/<variant>/seed_<k>/result.json (+ run.log, manifest.json).
"""
import concurrent.futures
import os
import time

import numpy as np

from evosynth.config import get_settings, with_overrides
from evosynth.core.individual import Individual
from evosynth.experiments.run_context import RunContext
from evosynth.seeding import stream


def _time(evaluator, vectors, workers):
    t = time.perf_counter()
    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            scores = list(ex.map(lambda v: evaluator.evaluate(Individual(v)), vectors))
    else:
        scores = [evaluator.evaluate(Individual(v)) for v in vectors]
    return time.perf_counter() - t, scores


def run_speedup(dataset="covertype", configurations=None, workers=None, variant="default", results_dir=None,
                subsample=None):
    with RunContext("speedup", dataset, variant, results_dir=results_dir) as run:
        return _run(run, dataset, configurations, workers, subsample)


def _run(run, dataset, configurations, workers, subsample):
    from evosynth.evaluators.tabular import TabularEvaluator

    s = get_settings()
    sp = s.experiments.speedup
    n = sp.configurations if configurations is None else configurations
    workers = (sp.workers or os.cpu_count() or 1) if workers is None else workers
    evaluators = {es: TabularEvaluator(dataset, subsample_size=subsample,
                                       settings=with_overrides(s, [f"evaluation.early_stopping={str(es).lower()}"]))
                  for es in (True, False)}
    dims = len(list(evaluators[True].space.keys()))
    rng = stream("speedup", "configurations")
    vectors = [rng.random(dims) for _ in range(n)]

    print("=" * 70)
    print(f" SPEED-UP BENCHMARK — {dataset}: {n} configurations, {workers} parallel workers")
    print("=" * 70)
    rows = {}
    for label, w, es in (("sequential, no early stopping", 1, False), ("sequential, early stopping", 1, True),
                         (f"parallel ({workers}), no early stopping", workers, False),
                         (f"parallel ({workers}), early stopping", workers, True)):
        seconds, scores = _time(evaluators[es], vectors, w)
        rows[label] = {"workers": w, "early_stopping": es, "seconds": round(seconds, 2),
                       "mean_cv_score": round(float(np.mean(scores)), 4)}
        print(f"  {label:<36} {seconds:8.1f} s   mean CV {np.mean(scores):.2f}%")
    ref = rows["sequential, no early stopping"]["seconds"]
    for r in rows.values():
        r["speedup"] = round(ref / max(r["seconds"], 1e-9), 2)
    print("\n  " + "  ".join(f"{k}: {v['speedup']}x" for k, v in rows.items()))
    result = {"dataset": dataset, "metric": evaluators[True].metric, "configurations": n, "workers": workers,
              "cpu_count": os.cpu_count(), "conditions": rows}
    run.write_result(result)
    return result
