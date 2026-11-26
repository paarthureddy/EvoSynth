"""
Budget-scaling runs with crash-safe checkpointing. Command: `evosynth scale`.

Runs EvoSynth at a given budget on a tabular dataset, writes an atomic checkpoint every
N evaluations, captures the full log and finishes with the post-search refit.

Settings: [experiments.scaling] (budgets, default_budget, pop_size, t1/k schedule,
checkpoint_interval), [datasets.<name>] (metric, split sizes), [selection], [refit], [evaluation].cv_folds.

Output: results/scaling/<dataset>/budget_<N>/seed_<k>/ (variant = budget_<N> unless --variant):
  run.log, events.jsonl, population.csv, checkpoint.json (every N evals), result.json, manifest.json
"""

import time

from evosynth import datasets
from evosynth.config import get_settings
from evosynth.core.decoding import decode_vector_to_dict
from evosynth.experiments.checkpointing import atomic_json_dump as save_atomic_checkpoint
from evosynth.experiments.run_context import RunContext


def schedule(budget, settings=None):
    """(t1_iters, k_iters) for a budget: max(min, int(fraction * budget)) from [experiments.scaling]."""
    sc = (settings or get_settings()).experiments.scaling
    return (max(sc.t1_iters_min, int(budget * sc.t1_iters_fraction)),
            max(sc.k_iters_min, int(budget * sc.k_iters_fraction)))


def run_benchmark(budget=None, pop_size=None, subsample=None, checkpoint_interval=None, variant=None,
                  results_dir=None, dataset="covertype"):
    """Arguments left as None come from [experiments.scaling] / [datasets.<name>]."""
    s = get_settings()
    sc = s.experiments.scaling
    budget = sc.default_budget if budget is None else budget
    if budget not in sc.budgets:
        raise ValueError(f"budget {budget} not in experiments.scaling.budgets {list(sc.budgets)}")
    pop_size = sc.pop_size if pop_size is None else pop_size
    checkpoint_interval = sc.checkpoint_interval if checkpoint_interval is None else checkpoint_interval
    subsample = datasets.get_spec(dataset).search_subsample if subsample is None else subsample

    with RunContext("scaling", dataset, variant or f"budget_{budget}", results_dir=results_dir) as run:
        _run_benchmark(run, s, budget, pop_size, subsample, checkpoint_interval, dataset)


def _run_benchmark(run, s, budget, pop_size, subsample, checkpoint_interval, dataset):
    from evosynth.engine.controller import DynamicOptimizer
    from evosynth.evaluators.tabular import TabularEvaluator
    from evosynth.search_spaces.tabular import get_tabular_search_space

    ckpt_path = run.file("checkpoint.json")
    start_time = time.time()
    print("=" * 80)
    print(f" EVOSYNTH SCALED EXPERIMENT — {datasets.get_spec(dataset).display_name} (BUDGET: {budget})")
    print(f" Population: {pop_size} | Checkpoint Interval: Every {checkpoint_interval} evals")
    print(f" Dataset Subsample: {subsample:,} ({s.evaluation.cv_folds}-Fold Stratified CV) | Latent Dims: "
          f"{len(list(get_tabular_search_space().keys()))}")
    print("=" * 80)

    space = get_tabular_search_space(include_model_type=True, include_fe_strategy=True)
    evaluator = TabularEvaluator(dataset, subsample_size=subsample)

    # Wrap evaluator to trigger atomic checkpoints every N evals
    original_evaluate = evaluator.evaluate
    eval_count = 0
    best_so_far = 0.0
    best_cfg = None

    def monitored_evaluate(individual):
        nonlocal eval_count, best_so_far, best_cfg
        score = original_evaluate(individual)
        eval_count += 1
        if score > best_so_far:
            best_so_far = score
            best_cfg = decode_vector_to_dict(individual.vector, space)

        if eval_count % checkpoint_interval == 0 or eval_count == budget:
            ckpt_data = {
                "budget": budget,
                "evals_used": eval_count,
                "best_cv_score": float(best_so_far),
                "best_config": best_cfg,
                "elapsed_seconds": round(time.time() - start_time, 2),
                "status": "in_progress" if eval_count < budget else "completed",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            save_atomic_checkpoint(ckpt_path, ckpt_data)
        return score

    evaluator.evaluate = monitored_evaluate

    # Determine tier iters based on budget
    t1_iters, k_iters = schedule(budget, s)

    optimizer = DynamicOptimizer(
        search_space=space,
        evaluator=evaluator,
        budget=budget,
        pop_size=pop_size,
        t1_iters=t1_iters,
        k_iters=k_iters,
        population_csv=run.file("population.csv"),
    )

    best = optimizer.run()

    # Confirm before select (Phase 3.7), then Step 6: refit on the held-out test rows
    from evosynth.experiments.selection import confirm_best
    from evosynth.models.refit import refit_and_score

    selected, winning_cfg, confirmation = confirm_best(evaluator, optimizer.evaluated, space)
    print("\n" + "=" * 80)
    print(" STEP 6: POST-SEARCH GENERALIZATION REFIT ON HELD-OUT TEST DATA")
    print("=" * 80)
    print("  Evaluating the selected architecture & hyperparameters at full capacity...")
    refit = refit_and_score(winning_cfg, dataset=dataset, phase="refit")
    m_type, fe_strat = refit["model_type"], refit["fe_strategy"]

    elapsed = time.time() - start_time
    print("\n  +----------------------------------------------------------------+")
    print(f"  | EVO-SYNTH FINAL BENCHMARK SUMMARY (BUDGET: {budget}, {refit['metric']})")
    print("  +----------------------------------------------------------------+")
    print(f"  | Total Search Time              : {elapsed/60:.1f} minutes")
    print(f"  | Search CV score (selected)     : {selected.fitness:.2f}% (best seen {best.fitness:.2f}%)")
    print(f"  | Held-out test score            : {refit['test_score']:.2f}% "
          f"({refit['train_rows']:,} train, {refit['test_rows']:,} test)")
    print(f"  | Optimal Architecture           : {m_type}")
    print(f"  | Optimal Feature Engineering    : {fe_strat}")
    print("  +----------------------------------------------------------------+")

    final_ckpt = {
        "budget": budget,
        "evals_used": budget,
        "metric": refit["metric"],
        "best_cv_score": float(best.fitness),
        "selected_cv_score": float(selected.fitness),
        "test_score": float(refit["test_score"]),
        "optimal_architecture": m_type,
        "optimal_fe_strategy": fe_strat,
        "selected_config": winning_cfg,
        "confirmation": confirmation,
        "failed_folds": evaluator.failed_folds,
        "elapsed_seconds": round(elapsed, 2),
        "status": "completed",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_atomic_checkpoint(ckpt_path, final_ckpt)
    run.write_result({k: v for k, v in final_ckpt.items() if k not in ("status", "timestamp")}
                     | {"winning_algorithm": selected.source_algorithm})

