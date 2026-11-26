"""
Competitor & floor baselines on any registered tabular dataset. Command: `evosynth baselines`.

  defaults       untuned XGBoost, LightGBM, RandomForest, ExtraTrees (library defaults), trained
                 on the refit train rows and scored on the held-out test rows (Phase 4.5)
  random_search  uniform random vectors in the same latent space as EvoSynth
  optuna_tpe     Optuna's TPE sampler over the same latent space (SQLite study, resumable)
  flaml          FLAML AutoML with a time budget (its own search and refit)

Random Search and Optuna are treated exactly like EvoSynth (Phase 4.4): the same evaluator
(same search rows, CV folds, early stopping, metric), the same confirm-before-select
([selection]) and the same refit (models/refit.py, phase "refit") of the configuration they
selected. Every method's inference cost is measured (models/inference.py).

Settings: [experiments.baselines], [selection], [refit], [inference], [datasets.<name>],
[run].seed (search) and data_seed (splits).

Output: one run folder per baseline, results/baselines/<dataset>/<baseline>/seed_<k>/
with run.log, report.txt, result.json, manifest.json (+ model.pkl, optuna.db, flaml.log).
"""
import json
import os
import time

import numpy as np

from evosynth.config import get_settings
from evosynth.core.individual import Individual
from evosynth.datasets import get_spec
from evosynth.experiments.run_context import RunContext
from evosynth.experiments.selection import confirm_best
from evosynth.models.factory import build_default_model, fit_model
from evosynth.models.inference import measure
from evosynth.models.metrics import score
from evosynth.models.refit import refit_and_score, refit_split
from evosynth.search_spaces.tabular import get_tabular_search_space
from evosynth.seeding import stream


def load_splits(dataset):
    """Same train / held-out test rows as EvoSynth's refit (datasets.partition)."""
    spec = get_spec(dataset, kind="tabular")
    print(f"  [DATA] {spec.display_name}: {spec.refit_train_rows:,} train / {spec.test_rows:,} test rows")
    return refit_split(dataset)


def _report(out_dir, title, lines):
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 70 + f"\n BASELINE REPORT: {title}\n" + "=" * 70 + "\n")
        for k, v in lines:
            f.write(f" {k:<22}: {v}\n")
        f.write("=" * 70 + "\n")


def _banner(text):
    print("\n" + "=" * 70 + f"\n {text}\n" + "=" * 70)


# ------------------------------------------------------------------------------ defaults
def run_defaults(out_dir, dataset="covertype"):
    """Untuned library defaults of every model family (Phase 4.5)."""
    _banner("BASELINE: UNTUNED LIBRARY DEFAULTS (XGBoost, LightGBM, RandomForest, ExtraTrees)")
    X_tr, X_te, y_tr, y_te = load_splits(dataset)
    spec = get_spec(dataset)
    models = {}
    for model_type in get_settings().experiments.baselines.default_models:
        t = time.time()
        model = build_default_model(model_type, xgb_eval_metric=spec.xgb_eval_metric)
        model = fit_model(model, model_type, X_tr, y_tr, phase="refit")
        fit_seconds = time.time() - t
        models[model_type] = {"test_score": score(model, X_te, y_te, spec.metric) * 100.0,
                              "fit_seconds": round(fit_seconds, 2), "inference": measure(model, X_te)}
        print(f"  [RESULT] {model_type:<13} {spec.metric} {models[model_type]['test_score']:.2f}% "
              f"(fit {fit_seconds:.1f}s, {models[model_type]['inference']['single_row_ms']:.2f} ms/row)")
    best = max(models, key=lambda m: models[m]["test_score"])
    _report(out_dir, "UNTUNED LIBRARY DEFAULTS",
            [(m, f"{r['test_score']:.2f}% test {spec.metric}") for m, r in models.items()]
            + [("Best", best), ("Train / test rows", f"{len(y_tr):,} / {len(y_te):,}")])
    return {"metric": spec.metric, "test_score": models[best]["test_score"], "model_type": best,
            "models": models}


# ------------------------------------------------------------------------------ tuned baselines
def _select_and_refit(out_dir, evaluator, individuals, space, dataset):
    """Confirm-before-select + shared refit, exactly as EvoSynth does after its search."""
    selected, cfg, confirmation = confirm_best(evaluator, individuals, space)
    refit = refit_and_score(cfg, dataset=dataset, verbose=False, model_path=os.path.join(out_dir, "model.pkl"))
    return selected, cfg, confirmation, refit


def _tuned_result(name, best_cv, selected, cfg, confirmation, refit, evaluator, n_trials, seconds):
    return {"metric": refit["metric"], "test_score": float(refit["test_score"]),
            "best_cv_score": float(best_cv), "selected_cv_score": float(selected.fitness),
            "model_type": refit["model_type"], "fe_strategy": refit["fe_strategy"],
            "selected_config": cfg, "confirmation": confirmation, "inference": refit["inference"],
            "trials": n_trials, "failed_folds": evaluator.failed_folds, "search_seconds": round(seconds, 2)}


def run_random_search(out_dir, n_trials=None, dataset="covertype"):
    n_trials = get_settings().experiments.baselines.random_search_trials if n_trials is None else n_trials
    _banner(f"BASELINE: RANDOM SEARCH ({n_trials} EVALUATIONS)")
    from evosynth.evaluators.tabular import TabularEvaluator

    space = get_tabular_search_space(include_model_type=True, include_fe_strategy=True)
    evaluator = TabularEvaluator(dataset)
    dims = len(list(space.keys()))
    rng = stream("baseline", "random_search")
    individuals = []
    start = time.time()
    for trial in range(1, n_trials + 1):
        ind = Individual(rng.random(dims))
        ind.set_fitness(evaluator.evaluate(ind))
        ind.source_algorithm = "RandomSearch"
        individuals.append(ind)
        if trial % 10 == 0 or trial == n_trials:
            print(f"  Trial {trial:>3}/{n_trials} | best CV {max(i.fitness for i in individuals):.2f}% "
                  f"| {time.time() - start:.1f}s")
    seconds = time.time() - start
    best_cv = max(i.fitness for i in individuals)
    selected, cfg, confirmation, refit = _select_and_refit(out_dir, evaluator, individuals, space, dataset)
    print(f"  [RESULT] Random Search test {refit['metric']}: {refit['test_score']:.2f}% ({refit['model_type']})")
    _report(out_dir, f"RANDOM SEARCH ({n_trials} TRIALS)",
            [("Best search CV", f"{best_cv:.2f}%"), ("Selected (confirmed)", f"{selected.fitness:.2f}%"),
             ("Held-out test", f"{refit['test_score']:.2f}% {refit['metric']}"),
             ("Model / FE", f"{refit['model_type']} / {refit['fe_strategy']}"),
             ("Search time", f"{seconds / 60:.1f} min")])
    return _tuned_result("random_search", best_cv, selected, cfg, confirmation, refit, evaluator, n_trials, seconds)


def run_optuna_baseline(out_dir, n_trials=None, dataset="covertype"):
    n_trials = get_settings().experiments.baselines.optuna_trials if n_trials is None else n_trials
    _banner(f"BASELINE: OPTUNA TPE SAMPLER ({n_trials} TRIALS)")
    try:
        import optuna
    except ImportError:
        print("  [!] Optuna not installed. Please run: pip install optuna")
        return {"skipped": "optuna not installed"}
    from evosynth.evaluators.tabular import TabularEvaluator

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    evaluator = TabularEvaluator(dataset)
    space = get_tabular_search_space(include_model_type=True, include_fe_strategy=True)
    dims = len(list(space.keys()))
    study = optuna.create_study(
        study_name=f"{dataset}_hpo", storage=f"sqlite:///{os.path.abspath(os.path.join(out_dir, 'optuna.db'))}",
        load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=get_settings().run.seed))
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - done)
    print(f"  [OPTUNA] Completed trials: {done}/{n_trials}. Running {remaining} more...")

    def objective(trial):
        return evaluator.evaluate(Individual(np.array([trial.suggest_float(f"x_{i}", 0.0, 1.0) for i in range(dims)])))

    start = time.time()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    seconds = time.time() - start

    individuals = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            ind = Individual(np.array([t.params[f"x_{i}"] for i in range(dims)]))
            ind.set_fitness(float(t.value))
            ind.source_algorithm = "OptunaTPE"
            individuals.append(ind)
    best_cv = study.best_value
    selected, cfg, confirmation, refit = _select_and_refit(out_dir, evaluator, individuals, space, dataset)
    print(f"  [RESULT] Optuna TPE test {refit['metric']}: {refit['test_score']:.2f}% ({refit['model_type']})")
    _report(out_dir, f"OPTUNA TPE ({n_trials} TRIALS)",
            [("Best search CV", f"{best_cv:.2f}%"), ("Selected (confirmed)", f"{selected.fitness:.2f}%"),
             ("Held-out test", f"{refit['test_score']:.2f}% {refit['metric']}"),
             ("Model / FE", f"{refit['model_type']} / {refit['fe_strategy']}"),
             ("Search time", f"{seconds / 60:.1f} min")])
    return _tuned_result("optuna_tpe", best_cv, selected, cfg, confirmation, refit, evaluator, n_trials, seconds)


def run_flaml_baseline(out_dir, time_budget=None, dataset="covertype"):
    bl = get_settings().experiments.baselines
    time_budget = bl.flaml_time_budget if time_budget is None else time_budget
    _banner(f"BASELINE: FLAML AUTOML (TIME BUDGET {time_budget}s)")
    try:
        from flaml import AutoML
    except ImportError:
        print("  [!] FLAML not installed. Please run: pip install flaml")
        return {"skipped": "flaml not installed"}

    X_tr, X_te, y_tr, y_te = load_splits(dataset)
    metric = get_spec(dataset).metric
    automl = AutoML()
    start = time.time()
    automl.fit(X_train=X_tr, y_train=y_tr, time_budget=time_budget, metric=metric, task="classification",
               log_file_name=os.path.join(out_dir, "flaml.log"), seed=get_settings().run.seed,
               estimator_list=list(bl.flaml_estimators))
    seconds = time.time() - start
    test = score(automl, X_te, y_te, metric) * 100.0
    try:
        inference = measure(automl, X_te)
    except Exception as e:                       # e.g. an AutoML object that cannot be pickled
        inference = {"error": f"{type(e).__name__}: {e}"}
    print(f"  [RESULT] FLAML test {metric}: {test:.2f}% ({automl.best_estimator})")
    _report(out_dir, "FLAML AUTOML", [("Time budget", f"{time_budget} s"), ("Best estimator", automl.best_estimator),
                                       ("Held-out test", f"{test:.2f}% {metric}"),
                                       ("Search time", f"{seconds / 60:.1f} min"),
                                       ("Best config", json.dumps(automl.best_config))])
    return {"metric": metric, "test_score": float(test), "model_type": str(automl.best_estimator),
            "best_config": automl.best_config, "inference": inference, "time_budget_seconds": time_budget,
            "search_seconds": round(seconds, 2)}


BASELINE_TESTS = ("all", "defaults", "random", "optuna", "flaml")

# --test name -> (run-folder name, function); Optuna resumes from its SQLite study
_BASELINES = {
    "defaults": ("defaults", lambda d, a: run_defaults(d, dataset=a["dataset"])),
    "random": ("random_search", lambda d, a: run_random_search(d, n_trials=a["trials"], dataset=a["dataset"])),
    "optuna": ("optuna_tpe", lambda d, a: run_optuna_baseline(d, n_trials=a["trials"], dataset=a["dataset"])),
    "flaml": ("flaml", lambda d, a: run_flaml_baseline(d, time_budget=a["time_budget"], dataset=a["dataset"])),
}


def run_baselines(test="all", trials=None, time_budget=None, results_dir=None, dataset="covertype"):
    """Run one or all baselines on `dataset`, each in its own run folder."""
    if test not in BASELINE_TESTS:
        raise ValueError(f"unknown baseline '{test}'. Choose from {BASELINE_TESTS}")
    get_spec(dataset, kind="tabular")            # fail early on an unknown dataset
    args = {"trials": trials, "time_budget": time_budget, "dataset": dataset}
    for name, (variant, fn) in _BASELINES.items():
        if test not in ("all", name):
            continue
        with RunContext("baselines", dataset, variant, results_dir=results_dir, resume=(name == "optuna")) as run:
            run.write_result(fn(run.path, args))
