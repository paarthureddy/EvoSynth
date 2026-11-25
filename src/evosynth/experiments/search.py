"""
Single EvoSynth search on a tabular dataset, followed by the post-search generalisation
refit (Step 6). Command: `evosynth run` (presets: search_demo [default], search_full).

Settings: [engine] (budget / pop_size / t1_iters / k_iters, set by the preset or --set),
[datasets.<name>] (search_subsample, display_name), [refit].

Output: results/search/<dataset>/<variant>/seed_<k>/ (variant = preset name unless --variant).
"""
import time

from evosynth import datasets
from evosynth.config import get_settings
from evosynth.experiments.run_context import RunContext


def run_search(dataset="covertype", mode_label="Fast Demo", subsample=None, refit=True,
               variant="search_demo", results_dir=None):
    """Run DynamicOptimizer with the active settings (inside a run folder); optionally refit the winner.
    Returns the best Individual."""
    with RunContext("search", dataset, variant, results_dir=results_dir) as run:
        return _run_search(run, dataset, mode_label, subsample, refit)


def _run_search(run, dataset, mode_label, subsample, refit):
    # Imported here so the controller is set up with the settings chosen by the CLI
    from evosynth.engine.controller import DynamicOptimizer
    from evosynth.evaluators.tabular import TabularEvaluator
    from evosynth.search_spaces.tabular import get_tabular_search_space

    s = get_settings()
    spec = datasets.get_spec(dataset)
    n = spec.search_subsample if subsample is None else subsample

    print("=" * 70)
    print(f" EVOSYNTH: REAL-WORLD BENCHMARK — {dataset.upper()} DATASET")
    print(f" Mode: {mode_label} | Budget: {s.engine.budget} | Pop: {s.engine.pop_size}")
    print(f" Dataset: {spec.display_name} → {n:,} stratified subsample")
    print("=" * 70)

    # 1. Search space with model switching (XGBoost / LightGBM / RandomForest / ExtraTrees) & generic FE
    space = get_tabular_search_space(include_model_type=True, include_fe_strategy=True)
    # 2. Real-world evaluator
    evaluator = TabularEvaluator(dataset, subsample_size=n)
    # 3. Dynamic Multi-Armed Bandit Meta-Optimizer (budget, pop_size, t1_iters, k_iters from [engine])
    optimizer = DynamicOptimizer(search_space=space, evaluator=evaluator,
                                 population_csv=run.file("population.csv"))
    t0 = time.time()
    best = optimizer.run()
    from evosynth.core.decoding import decode_vector_to_dict
    result = {
        "metric": "cv_accuracy",
        "best_cv_score": float(best.fitness),
        "best_config": decode_vector_to_dict(best.vector, space),
        "winning_algorithm": best.source_algorithm,
        "evals_used": optimizer.evals_used,
        "search_seconds": round(time.time() - t0, 2),
        "refit": None,
    }

    if not refit:
        run.write_result(result)
        return best

    # 4. Generalization evaluation on the held-out refit split
    r = s.refit
    print("\n" + "=" * 80)
    print(" STEP 6: POST-SEARCH GENERALIZATION REFIT ON HELD-OUT TEST DATA")
    print("=" * 80)
    print("  Evaluating winning architecture & hyperparameters at full capacity...")
    try:
        from evosynth.models.refit import refit_and_score

        refit_res = refit_and_score(result["best_config"], dataset=dataset, phase="refit")
        test_acc, m_type, fe_strat = refit_res["test_accuracy"], refit_res["model_type"], refit_res["fe_strategy"]
        result["refit"] = {"test_accuracy": float(test_acc), "model_type": m_type, "fe_strategy": fe_strat,
                           "train_rows": r.train_rows, "test_rows": r.test_rows}

        print(f"\n  +----------------------------------------------------------------+")
        print(f"  | EVO-SYNTH FINAL BENCHMARK SUMMARY                             |")
        print(f"  +----------------------------------------------------------------+")
        print(f"  | Search Phase CV Accuracy       : {best.fitness:.2f}% (Search slice)")
        print(f"  | Generalization Test Accuracy   : {test_acc:.2f}% "
              f"({r.train_rows // 1000}k train, {r.test_rows // 1000}k test)")
        print(f"  | Optimal Architecture           : {m_type}")
        print(f"  | Optimal Feature Engineering    : {fe_strat}")
        print(f"  +----------------------------------------------------------------+")
    except Exception as e:
        print(f"  Refit skipped: {e}")
        result["refit"] = {"error": f"{type(e).__name__}: {e}"}

    run.write_result(result)
    return best
