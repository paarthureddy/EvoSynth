"""
Budget-scaling runs with crash-safe checkpointing. Command: `evosynth scale`.

Runs EvoSynth at a given budget on a tabular dataset, writes an atomic checkpoint every
N evaluations, captures the full log and finishes with the post-search refit.

Settings: [experiments.scaling] (budgets, default_budget, pop_size, t1/k schedule,
checkpoint_interval), [datasets.<name>], [refit], [evaluation].cv_folds.

Outputs:
  - results_forest/[<run_name>/]evosynth/run_<budget>.txt
  - results_forest/[<run_name>/]evosynth/checkpoints/checkpoint_<budget>.json
"""

import os
import sys
import time
import json

from evosynth import datasets
from evosynth.config import get_settings
from evosynth.core.decoding import decode_vector_to_dict
from evosynth.experiments.checkpointing import TeeLogger, atomic_json_dump as save_atomic_checkpoint
from evosynth.paths import PROJECT_ROOT
_ROOT_DIR = str(PROJECT_ROOT)


def schedule(budget, settings=None):
    """(t1_iters, k_iters) for a budget: max(min, int(fraction * budget)) from [experiments.scaling]."""
    sc = (settings or get_settings()).experiments.scaling
    return (max(sc.t1_iters_min, int(budget * sc.t1_iters_fraction)),
            max(sc.k_iters_min, int(budget * sc.k_iters_fraction)))


def run_benchmark(budget=None, pop_size=None, subsample=None, checkpoint_interval=None, run_name=None,
                  output_dir=None, dataset="covertype"):
    """Arguments left as None come from [experiments.scaling] / [datasets.<name>]."""
    from evosynth.engine.controller import DynamicOptimizer
    from evosynth.evaluators.tabular import TabularEvaluator
    from evosynth.search_spaces.tabular import get_tabular_search_space

    s = get_settings()
    sc = s.experiments.scaling
    budget = sc.default_budget if budget is None else budget
    if budget not in sc.budgets:
        raise ValueError(f"budget {budget} not in experiments.scaling.budgets {list(sc.budgets)}")
    pop_size = sc.pop_size if pop_size is None else pop_size
    checkpoint_interval = sc.checkpoint_interval if checkpoint_interval is None else checkpoint_interval
    subsample = datasets.get_spec(dataset).search_subsample if subsample is None else subsample

    if output_dir:
        target_dir = output_dir
    elif run_name:
        target_dir = os.path.join(_ROOT_DIR, "results_forest", run_name, "evosynth")
    else:
        target_dir = os.path.join(_ROOT_DIR, "results_forest", "evosynth")

    ckpt_dir = os.path.join(target_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    log_path = os.path.join(target_dir, f"run_{budget}.txt")
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{budget}.json")

    tee = TeeLogger(log_path, sys.stdout)
    sys.stdout = tee

    start_time = time.time()
    try:
        print("=" * 80)
        print(f" EVOSYNTH SCALED EXPERIMENT — UCI FOREST COVERTYPE (BUDGET: {budget})")
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
                    "best_cv_accuracy": float(best_so_far),
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
            k_iters=k_iters
        )

        best = optimizer.run()

        # Step 6: Full Generalization Refit on 60k / 20k unseen test data
        print("\n" + "=" * 80)
        print(" STEP 6: POST-SEARCH GENERALIZATION REFIT ON HELD-OUT TEST DATA")
        print("=" * 80)
        print("  Evaluating winning architecture & hyperparameters at full capacity...")

        from evosynth.models.refit import refit_and_score

        winning_cfg = decode_vector_to_dict(best.vector, space)
        refit = refit_and_score(winning_cfg, dataset=dataset, phase="refit")
        test_acc, m_type, fe_strat = refit["test_accuracy"], refit["model_type"], refit["fe_strategy"]

        elapsed = time.time() - start_time
        print("\n  +----------------------------------------------------------------+")
        print(f"  | EVO-SYNTH FINAL BENCHMARK SUMMARY (BUDGET: {budget})          |")
        print("  +----------------------------------------------------------------+")
        print(f"  | Total Search Time              : {elapsed/60:.1f} minutes")
        print(f"  | Search Phase CV Accuracy       : {best.fitness:.2f}% (20k slice)")
        print(f"  | Generalization Test Accuracy   : {test_acc:.2f}% (60k train, 20k test)")
        print(f"  | Optimal Architecture           : {m_type}")
        print(f"  | Optimal Feature Engineering    : {fe_strat}")
        print("  +----------------------------------------------------------------+")

        final_ckpt = {
            "budget": budget,
            "evals_used": budget,
            "best_cv_accuracy": float(best.fitness),
            "generalization_test_accuracy": float(test_acc),
            "optimal_architecture": m_type,
            "optimal_fe_strategy": fe_strat,
            "best_config": winning_cfg,
            "elapsed_seconds": round(elapsed, 2),
            "status": "completed",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        save_atomic_checkpoint(ckpt_path, final_ckpt)

    finally:
        sys.stdout = tee.stream
        tee.close()

