"""
EvoSynth Real-World Benchmark: UCI Forest Covertype Dataset
============================================================
Runs the full Dynamic Multi-Armed Bandit Meta-Optimizer on the Forest Covertype
dataset (581,012 samples, 54 features, 7-class multiclass classification).

Usage:
    python scripts/run_covertype.py              # Fast demo (120 evals, ~3-5 min)
    python scripts/run_covertype.py --full       # Full benchmark (500 evals, ~15-20 min)
"""
import os
import sys
import argparse

from evosynth.paths import PROJECT_ROOT
_ROOT_DIR = str(PROJECT_ROOT)


from evosynth.search_spaces.tabular import get_xgboost_search_space
from evosynth.evaluators.tabular import TabularEvaluator
from evosynth.engine.controller import DynamicOptimizer


def main():
    parser = argparse.ArgumentParser(description="EvoSynth Covertype Real-World Benchmark")
    parser.add_argument("--budget", type=int, default=120, help="Total evaluation budget")
    parser.add_argument("--pop_size", type=int, default=20, help="Population size")
    parser.add_argument("--t1_iters", type=int, default=3, help="Tier 1 competition iterations")
    parser.add_argument("--k_iters", type=int, default=2, help="Checkpoint interval")
    parser.add_argument("--subsample", type=int, default=20000, help="Stratified subsample size")
    parser.add_argument("--full", action="store_true", help="Run full benchmark (500 evals, pop=40)")
    args = parser.parse_args()

    if args.full:
        budget = 500
        pop_size = 40
        t1_iters = 5
        k_iters = 3
    else:
        budget = args.budget
        pop_size = args.pop_size
        t1_iters = args.t1_iters
        k_iters = args.k_iters

    print("=" * 70)
    print(" EVOSYNTH: REAL-WORLD BENCHMARK — UCI FOREST COVERTYPE DATASET")
    print(f" Mode: {'Full Benchmark' if args.full else 'Fast Demo'} | Budget: {budget} | Pop: {pop_size}")
    print(f" Dataset: 581,012 samples → {args.subsample:,} stratified subsample | 54 features | 7 classes")
    print("=" * 70)

    # 1. Search space with model switching (XGBoost / LightGBM / RandomForest / ExtraTrees) & generic FE
    space = get_xgboost_search_space(include_model_type=True, include_fe_strategy=True)

    # 2. Real-world evaluator on Forest Covertype
    evaluator = TabularEvaluator("covertype", subsample_size=args.subsample)

    # 3. Dynamic Multi-Armed Bandit Meta-Optimizer
    optimizer = DynamicOptimizer(
        search_space=space,
        evaluator=evaluator,
        budget=budget,
        pop_size=pop_size,
        t1_iters=t1_iters,
        k_iters=k_iters
    )

    best = optimizer.run()

    # 4. Generalization Evaluation on Expanded Unseen Held-out Split
    print("\n" + "=" * 80)
    print(" STEP 6: POST-SEARCH GENERALIZATION REFIT ON HELD-OUT TEST DATA")
    print("=" * 80)
    print("  Evaluating winning architecture & hyperparameters at full capacity...")
    try:
        from evosynth.core.decoding import decode_vector_to_dict
        from evosynth.models.refit import refit_and_score

        refit = refit_and_score(decode_vector_to_dict(best.vector, space), dataset="covertype", phase="refit")
        test_acc, m_type, fe_strat = refit["test_accuracy"], refit["model_type"], refit["fe_strategy"]

        print(f"\n  +----------------------------------------------------------------+")
        print(f"  | EVO-SYNTH FINAL BENCHMARK SUMMARY                             |")
        print(f"  +----------------------------------------------------------------+")
        print(f"  | Search Phase CV Accuracy       : {best.fitness:.2f}% (Search slice)")
        print(f"  | Generalization Test Accuracy   : {test_acc:.2f}% (60k train, 20k test)")
        print(f"  | Optimal Architecture           : {m_type}")
        print(f"  | Optimal Feature Engineering    : {fe_strat}")
        print(f"  +----------------------------------------------------------------+")
    except Exception as e:
        print(f"  Refit skipped: {e}")

    return best


if __name__ == "__main__":
    main()

