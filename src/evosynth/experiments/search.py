"""
Single EvoSynth search on a tabular dataset, followed by the post-search generalisation
refit (Step 6). Command: `evosynth run` (presets: search_demo [default], search_full).

Settings: [engine] (budget / pop_size / t1_iters / k_iters, set by the preset or --set),
[datasets.<name>] (search_subsample, display_name), [refit].
"""
from evosynth import datasets
from evosynth.config import get_settings


def run_search(dataset="covertype", mode_label="Fast Demo", subsample=None, refit=True):
    """Run DynamicOptimizer with the active settings; optionally refit the winner. Returns the best Individual."""
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
    optimizer = DynamicOptimizer(search_space=space, evaluator=evaluator)
    best = optimizer.run()

    if not refit:
        return best

    # 4. Generalization evaluation on the held-out refit split
    r = s.refit
    print("\n" + "=" * 80)
    print(" STEP 6: POST-SEARCH GENERALIZATION REFIT ON HELD-OUT TEST DATA")
    print("=" * 80)
    print("  Evaluating winning architecture & hyperparameters at full capacity...")
    try:
        from evosynth.core.decoding import decode_vector_to_dict
        from evosynth.models.refit import refit_and_score

        result = refit_and_score(decode_vector_to_dict(best.vector, space), dataset=dataset, phase="refit")
        test_acc, m_type, fe_strat = result["test_accuracy"], result["model_type"], result["fe_strategy"]

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

    return best
