"""
Single EvoSynth search on a tabular dataset, followed by the post-search generalisation
refit (Step 6). Command: `evosynth run` (presets: search_demo [default], search_full).

Settings: [engine] (budget / pop_size / t1_iters / k_iters, set by the preset or --set),
[datasets.<name>] (metric, split sizes, display_name), [selection], [refit].

Output: results/search/<dataset>/<variant>/seed_<k>/ (variant = preset name unless --variant).
"""
import time

from evosynth import datasets
from evosynth.config import get_settings
from evosynth.experiments.run_context import RunContext


def run_search(dataset="covertype", mode_label="Fast Demo", subsample=None, refit=True,
               variant="search_demo", results_dir=None):
    """Run DynamicOptimizer with the active settings (inside a run folder); optionally refit the winner.
    Returns the selected Individual (after confirm-before-select)."""
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
    search_seconds = round(time.time() - t0, 2)

    # 4. Confirm before select (Phase 3.7): re-score the top configurations with more folds
    from evosynth.experiments.selection import confirm_best
    selected, selected_cfg, confirmation = confirm_best(evaluator, optimizer.evaluated, space)
    metric = evaluator.metric
    result = {
        "metric": metric,
        "best_cv_score": float(best.fitness),
        "selected_cv_score": float(selected.fitness),
        "selected_config": selected_cfg,
        "winning_algorithm": selected.source_algorithm,
        "confirmation": confirmation,
        "failed_folds": evaluator.failed_folds,
        "evals_used": optimizer.evals_used,
        "search_seconds": search_seconds,
        "refit": None,
    }

    if not refit:
        run.write_result(result)
        return selected

    # 5. Generalization evaluation on the held-out test rows
    print("\n" + "=" * 80)
    print(" STEP 6: POST-SEARCH GENERALIZATION REFIT ON HELD-OUT TEST DATA")
    print("=" * 80)
    print("  Evaluating the selected architecture & hyperparameters at full capacity...")
    try:
        from evosynth.models.refit import refit_and_score

        refit_res = refit_and_score(selected_cfg, dataset=dataset, model_path=run.file("model.pkl"))
        result["refit"] = {k: refit_res[k] for k in ("test_score", "metric", "model_type", "fe_strategy",
                                                      "train_rows", "test_rows", "inference")}
        conf = ""
        if confirmation:
            picked = confirmation["candidates"][confirmation["selected_search_rank"] - 1]
            conf = f" / confirmed {picked['confirmed_score']:.2f}% ({confirmation['confirm_folds']}-fold)"
        print(f"\n  +----------------------------------------------------------------+")
        print(f"  | EVO-SYNTH FINAL BENCHMARK SUMMARY ({metric})")
        print(f"  +----------------------------------------------------------------+")
        print(f"  | Search CV score (selected)     : {selected.fitness:.2f}%{conf}")
        print(f"  | Held-out test score            : {refit_res['test_score']:.2f}% "
              f"({refit_res['train_rows']:,} train, {refit_res['test_rows']:,} test)")
        print(f"  | Optimal Architecture           : {refit_res['model_type']}")
        print(f"  | Optimal Feature Engineering    : {refit_res['fe_strategy']}")
        inf = refit_res["inference"]
        print(f"  | Inference (measured)           : {inf['single_row_ms']:.2f} ms / row, "
              f"{inf['batch_rows_per_s']:,.0f} rows/s in batch, model {inf['model_mb']:.1f} MB")
        print(f"  +----------------------------------------------------------------+")
    except Exception as e:
        print(f"  Refit skipped: {e}")
        result["refit"] = {"error": f"{type(e).__name__}: {e}"}

    run.write_result(result)
    return selected
