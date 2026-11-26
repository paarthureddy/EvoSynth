"""DynamicOptimizer end to end on the toy objective (no model training)."""
import os

import numpy as np

from evosynth.engine.controller import DynamicOptimizer
from evosynth.engine.scoring import CompositeScoringEngine


def _run(space, evaluator, budget=120, pop_size=40, **kw):
    opt = DynamicOptimizer(search_space=space, evaluator=evaluator, budget=budget, pop_size=pop_size,
                           t1_iters=2, k_iters=1, max_workers=1, **kw)
    return opt, opt.run()


def test_run_uses_exactly_the_budget(toy_problem):
    np.random.seed(0)
    opt, best = _run(*toy_problem)
    assert opt.evals_used == 120
    assert len(opt.history) == 120
    assert best.fitness == max(h["fitness"] for h in opt.history)
    assert best.source_algorithm


def test_best_so_far_is_monotonic(toy_problem):
    np.random.seed(1)
    opt, _ = _run(*toy_problem)
    trace = [h["best_so_far"] for h in opt.history]
    assert trace == sorted(trace)


def test_population_csv_has_one_row_per_evaluation(toy_problem, tmp_path):
    np.random.seed(2)
    csv = tmp_path / "population.csv"
    _run(*toy_problem, population_csv=str(csv))
    lines = csv.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("Eval_ID,Phase,Algorithm,Fitness,")
    assert len(lines) == 1 + 120
    assert {l.split(",")[1] for l in lines[1:]} <= {"Init", "T1", "T2"}


def test_import_has_no_side_effects(tmp_path):
    # importing the engine must not create files or replace stdout (R5)
    import importlib
    import sys
    import evosynth.engine.controller as c
    before = sys.stdout
    importlib.reload(c)
    assert sys.stdout is before


def test_composite_scores_are_bounded():
    eng = CompositeScoringEngine()
    raw = {a: {"fitness": f, "speed": f / 10, "efficiency": 1.0, "diversity": 0.5, "trend": f / 100}
           for a, f in zip(["GA", "PSO", "DE", "CMAES"], [80.0, 82.0, 85.0, 81.0])}
    scores = eng.calculate_scores(raw)
    assert set(scores) == set(raw)
    assert all(0.0 <= v <= 1.0 + 1e-9 for v in scores.values())
    assert max(scores, key=scores.get) == "DE"
