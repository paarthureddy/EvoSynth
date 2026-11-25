"""
EvoSynth Scaled Covertype Benchmark Runner
===========================================
Runs EvoSynth at scaled budgets (e.g., 300, 500) on the UCI Forest Covertype dataset.
Includes atomic checkpointing every 10 evaluations, full log capture, and post-search
generalization refit on 60,000 train / 20,000 held-out test samples.

Outputs:
  - results_forest/evosynth/run_<budget>.txt
  - results_forest/evosynth/checkpoints/checkpoint_<budget>.json
"""

import os
import sys
import time
import json
import argparse
import numpy as np

from evosynth.paths import PROJECT_ROOT
_ROOT_DIR = str(PROJECT_ROOT)


from evosynth.search_spaces.tabular import get_xgboost_search_space
from evosynth.evaluators.tabular import TabularEvaluator
from evosynth.engine.controller import DynamicOptimizer
from evosynth.core.decoding import decode_vector_to_dict
from evosynth.experiments.checkpointing import TeeLogger, atomic_json_dump as save_atomic_checkpoint
from ConfigSpace import Configuration
import ConfigSpace.hyperparameters as CSH


def run_benchmark(budget=300, pop_size=30, subsample=20000, checkpoint_interval=10, run_name=None, output_dir=None):
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
        print(f" Dataset Subsample: {subsample:,} (2-Fold Stratified CV) | Latent Dims: 9")
        print("=" * 80)

        space = get_xgboost_search_space(include_model_type=True, include_fe_strategy=True)
        evaluator = TabularEvaluator("covertype", subsample_size=subsample)

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
        t1_iters = max(3, int(budget * 0.03))
        k_iters = max(2, int(budget * 0.02))

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
        refit = refit_and_score(winning_cfg, dataset="covertype", phase="refit")
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


def main():
    parser = argparse.ArgumentParser(description="EvoSynth Scaled Covertype Benchmark")
    parser.add_argument("--budget", type=int, default=300, choices=[120, 300, 500, 1000],
                        help="Evaluation budget (default: 300)")
    parser.add_argument("--pop_size", type=int, default=30, help="Population size (default: 30)")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                        help="Save checkpoint every N evals (default: 10)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Daily run folder name (e.g. Test1_19_Nov)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Direct path to output directory")
    args = parser.parse_args()

    run_benchmark(
        budget=args.budget,
        pop_size=args.pop_size,
        checkpoint_interval=args.checkpoint_interval,
        run_name=args.run_name,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
