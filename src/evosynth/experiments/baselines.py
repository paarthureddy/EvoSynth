"""
Competitor & Floor Baselines Runner — UCI Forest Covertype
===========================================================
Command: `evosynth baselines`. Settings: [experiments.baselines], [refit], [run].random_state.
Runs and logs the baseline comparisons on the exact same refit train / test split:
  1. Default ExtraTrees Classifier (Untuned Floor)
  2. Random Search (120 evaluations across the 9-dimensional search space)
  3. Optuna TPE (120 trials, with SQLite persistence for auto-resume)
  4. FLAML AutoML (Time-budget matched to EvoSynth's ~10 min runtime)

Output: one run folder per baseline, results/baselines/<dataset>/<baseline>/seed_<k>/
(baseline = default_extratrees | random_search | optuna_tpe | flaml), each with run.log,
report.txt, result.json, manifest.json (+ optuna.db / flaml.log).
"""

import os
import sys
import time
import json
import argparse
import numpy as np

from evosynth.experiments.run_context import RunContext


from sklearn.ensemble import ExtraTreesClassifier

from evosynth.config import get_settings
from evosynth.core.decoding import decode_vector_to_dict
from evosynth.models.refit import refit_and_score, refit_split


def load_covertype_splits():
    """Same stratified refit split as EvoSynth ([refit] train_rows / test_rows)."""
    r = get_settings().refit
    print(f"  [DATA] Loading Covertype dataset ({r.train_rows:,} train / {r.test_rows:,} test)...")
    return refit_split("covertype")


def run_default_extratrees(out_dir):
    log_file = os.path.join(out_dir, "report.txt")
    print("\n" + "=" * 70)
    print(" RUNNING BASELINE 1: DEFAULT EXTRATREES (UNTUNED FLOOR)")
    print("=" * 70)
    X_tr, X_te, y_tr, y_te = load_covertype_splits()

    start = time.time()
    s = get_settings()
    model = ExtraTreesClassifier(n_estimators=s.experiments.baselines.default_extratrees_trees,
                                 random_state=s.run.random_state, n_jobs=-1)
    model.fit(X_tr, y_tr)
    fit_time = time.time() - start

    preds = model.predict(X_te)
    acc = float(np.mean(preds == y_te)) * 100.0

    print(f"  [RESULT] Default ExtraTrees Accuracy : {acc:.2f}%")
    print(f"  [TIME]   Training Time               : {fit_time:.1f}s")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(" BASELINE REPORT: DEFAULT EXTRATREES (UNTUNED)\n")
        f.write("=" * 70 + "\n")
        f.write(f" Model                : ExtraTreesClassifier (sklearn defaults)\n")
        f.write(f" Train Split          : 60,000 samples\n")
        f.write(f" Test Split           : 20,000 samples (held-out)\n")
        f.write(f" Accuracy             : {acc:.2f}%\n")
        f.write(f" Fit Duration         : {fit_time:.1f} seconds\n")
        f.write("=" * 70 + "\n")
    return {"metric": "test_accuracy", "test_accuracy": acc, "model_type": "ExtraTrees",
            "n_estimators": s.experiments.baselines.default_extratrees_trees, "fit_seconds": round(fit_time, 2)}


def run_random_search(out_dir, n_trials=None):
    n_trials = get_settings().experiments.baselines.random_search_trials if n_trials is None else n_trials
    log_file = os.path.join(out_dir, "report.txt")
    print("\n" + "=" * 70)
    print(f" RUNNING BASELINE 2: RANDOM SEARCH ({n_trials} EVALUATIONS)")
    print("=" * 70)

    from evosynth.search_spaces.tabular import get_xgboost_search_space
    from evosynth.evaluators.tabular import TabularEvaluator
    from evosynth.core.individual import Individual

    space = get_xgboost_search_space(include_model_type=True, include_fe_strategy=True)
    evaluator = TabularEvaluator("covertype")
    dims = len(list(space.keys()))

    best_acc = 0.0
    best_vec = None
    start = time.time()

    for trial in range(1, n_trials + 1):
        vec = np.random.rand(dims)
        ind = Individual(vec)
        score = evaluator.evaluate(ind)
        if score > best_acc:
            best_acc = score
            best_vec = vec
        if trial % 10 == 0 or trial == n_trials:
            print(f"  Trial {trial:>3}/{n_trials} | Best CV Accuracy: {best_acc:.2f}% | Elapsed: {(time.time()-start):.1f}s")

    # Refit winning vector on the shared refit split (library-default model of the chosen family)
    load_covertype_splits()
    best_cfg = decode_vector_to_dict(best_vec, space)
    refit = refit_and_score(best_cfg, dataset="covertype", phase="baseline_default", verbose=False)
    test_acc, m_type, fe_strat = refit["test_accuracy"], refit["model_type"], refit["fe_strategy"]

    total_time = time.time() - start
    print(f"  [RESULT] Random Search Test Accuracy: {test_acc:.2f}% (Model: {m_type})")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f" BASELINE REPORT: RANDOM SEARCH ({n_trials} TRIALS)\n")
        f.write("=" * 70 + "\n")
        f.write(f" Search Phase CV Acc  : {best_acc:.2f}%\n")
        f.write(f" Refit Test Accuracy  : {test_acc:.2f}%\n")
        f.write(f" Discovered Model     : {m_type}\n")
        f.write(f" Discovered FE        : {fe_strat}\n")
        f.write(f" Total Runtime        : {total_time/60:.1f} minutes\n")
        f.write("=" * 70 + "\n")
    return {"metric": "test_accuracy", "test_accuracy": float(test_acc), "best_cv_accuracy": float(best_acc),
            "model_type": m_type, "fe_strategy": fe_strat, "best_config": best_cfg, "trials": n_trials,
            "total_seconds": round(total_time, 2)}


def run_optuna_baseline(out_dir, n_trials=None):
    n_trials = get_settings().experiments.baselines.optuna_trials if n_trials is None else n_trials
    log_file = os.path.join(out_dir, "report.txt")
    db_file = os.path.join(out_dir, "optuna.db")
    print("\n" + "=" * 70)
    print(f" RUNNING BASELINE 3: OPTUNA TPE SAMPLER ({n_trials} TRIALS)")
    print("=" * 70)

    try:
        import optuna
    except ImportError:
        print("  [!] Optuna not installed. Please run: pip install optuna")
        return {"skipped": "optuna not installed"}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from evosynth.evaluators.tabular import TabularEvaluator
    from evosynth.search_spaces.tabular import get_xgboost_search_space
    from evosynth.core.individual import Individual

    evaluator = TabularEvaluator("covertype")
    space = get_xgboost_search_space(include_model_type=True, include_fe_strategy=True)
    dims = len(list(space.keys()))

    storage_url = f"sqlite:///{os.path.abspath(db_file)}"
    study = optuna.create_study(
        study_name="covertype_hpo",
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=get_settings().run.random_state)
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)
    print(f"  [OPTUNA] Prior completed trials: {completed}/{n_trials}. Running remaining: {remaining}...")

    def objective(trial):
        vec = np.array([trial.suggest_float(f"x_{i}", 0.0, 1.0) for i in range(dims)])
        ind = Individual(vec)
        return evaluator.evaluate(ind)

    start = time.time()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    search_time = time.time() - start

    best_cv = study.best_value
    best_vec = np.array([study.best_params[f"x_{i}"] for i in range(dims)])

    best_cfg = decode_vector_to_dict(best_vec, space)
    load_covertype_splits()
    refit = refit_and_score(best_cfg, dataset="covertype", phase="baseline_default", verbose=False)
    test_acc, m_type, fe_strat = refit["test_accuracy"], refit["model_type"], refit["fe_strategy"]

    print(f"  [RESULT] Optuna TPE Test Accuracy: {test_acc:.2f}% (CV: {best_cv:.2f}%)")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f" BASELINE REPORT: OPTUNA TPE SAMPLER ({n_trials} TRIALS)\n")
        f.write("=" * 70 + "\n")
        f.write(f" Search Phase CV Acc  : {best_cv:.2f}%\n")
        f.write(f" Refit Test Accuracy  : {test_acc:.2f}%\n")
        f.write(f" Discovered Model     : {m_type}\n")
        f.write(f" Discovered FE        : {fe_strat}\n")
        f.write(f" Search Runtime       : {search_time/60:.1f} minutes\n")
        f.write("=" * 70 + "\n")
    return {"metric": "test_accuracy", "test_accuracy": float(test_acc), "best_cv_accuracy": float(best_cv),
            "model_type": m_type, "fe_strategy": fe_strat, "best_config": best_cfg, "trials": n_trials,
            "search_seconds": round(search_time, 2)}


def run_flaml_baseline(out_dir, time_budget=None):
    bl = get_settings().experiments.baselines
    time_budget = bl.flaml_time_budget if time_budget is None else time_budget
    log_file = os.path.join(out_dir, "report.txt")
    print("\n" + "=" * 70)
    print(f" RUNNING BASELINE 4: FLAML AUTOML (TIME BUDGET: {time_budget}s / ~10m)")
    print("=" * 70)

    try:
        from flaml import AutoML
    except ImportError:
        print("  [!] FLAML not installed. Please run: pip install flaml")
        return {"skipped": "flaml not installed"}

    X_tr, X_te, y_tr, y_te = load_covertype_splits()
    automl = AutoML()
    settings = {
        "time_budget": time_budget,
        "metric": bl.flaml_metric,
        "task": "classification",
        "log_file_name": os.path.join(out_dir, "flaml.log"),
        "seed": get_settings().run.random_state,
        "estimator_list": list(bl.flaml_estimators)
    }

    start = time.time()
    automl.fit(X_train=X_tr, y_train=y_tr, **settings)
    total_time = time.time() - start

    preds = automl.predict(X_te)
    test_acc = float(np.mean(preds == y_te)) * 100.0

    print(f"  [RESULT] FLAML Test Accuracy : {test_acc:.2f}%")
    print(f"  [RESULT] FLAML Best Model    : {automl.best_estimator}")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f" BASELINE REPORT: FLAML AUTOML\n")
        f.write("=" * 70 + "\n")
        f.write(f" Time Budget Allotted : {time_budget} seconds\n")
        f.write(f" Best Estimator Found : {automl.best_estimator}\n")
        f.write(f" Test Accuracy        : {test_acc:.2f}%\n")
        f.write(f" Actual Search Time   : {total_time/60:.1f} minutes\n")
        f.write(f" Best Config          : {json.dumps(automl.best_config, indent=2)}\n")
        f.write("=" * 70 + "\n")
    return {"metric": "test_accuracy", "test_accuracy": float(test_acc), "model_type": str(automl.best_estimator),
            "best_config": automl.best_config, "time_budget_seconds": time_budget,
            "total_seconds": round(total_time, 2)}


BASELINE_TESTS = ("all", "extratrees", "random", "optuna", "flaml")


# --test name -> (run-folder name, function); Optuna resumes from its SQLite study
_BASELINES = {
    "extratrees": ("default_extratrees", lambda d, a: run_default_extratrees(d)),
    "random": ("random_search", lambda d, a: run_random_search(d, n_trials=a["trials"])),
    "optuna": ("optuna_tpe", lambda d, a: run_optuna_baseline(d, n_trials=a["trials"])),
    "flaml": ("flaml", lambda d, a: run_flaml_baseline(d, time_budget=a["time_budget"])),
}


def run_baselines(test="all", trials=None, time_budget=None, results_dir=None):
    """Run one or all baselines, each in its own run folder (Covertype only until Phase 3)."""
    if test not in BASELINE_TESTS:
        raise ValueError(f"unknown baseline '{test}'. Choose from {BASELINE_TESTS}")
    args = {"trials": trials, "time_budget": time_budget}
    for name, (variant, fn) in _BASELINES.items():
        if test not in ("all", name):
            continue
        with RunContext("baselines", "covertype", variant, results_dir=results_dir, resume=(name == "optuna")) as run:
            run.write_result(fn(run.path, args))
