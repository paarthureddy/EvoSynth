import os
import sys
import argparse
import numpy as np
from ConfigSpace import Configuration

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from automl_engine.core.search_space import get_xgboost_search_space
from automl_engine.core.evaluator import XGBoostEvaluator
from automl_engine.engine.controller import DynamicOptimizer


def main():
    parser = argparse.ArgumentParser(description="EvoSynth Dynamic Multi-Model AutoML")
    parser.add_argument("--budget", type=int, default=120, help="Total evaluation budget (default: 120 for fast demo, 2500 for full run)")
    parser.add_argument("--pop_size", type=int, default=20, help="Population size (default: 20 for fast demo, 100 for full run)")
    parser.add_argument("--t1_iters", type=int, default=3, help="Tier 1 competition iterations (default: 3)")
    parser.add_argument("--k_iters", type=int, default=2, help="Checkpoint interval (default: 2)")
    parser.add_argument("--full", action="store_true", help="Run full benchmark with 2500 budget and 100 population")
    args = parser.parse_args()

    if args.full:
        budget = 2500
        pop_size = 100
        t1_iters = 10
        k_iters = 5
    else:
        budget = args.budget
        pop_size = args.pop_size
        t1_iters = args.t1_iters
        k_iters = args.k_iters

    print("="*60)
    print(" EVOSYNTH: DYNAMIC MULTI-MODEL AUTOML (XGBOOST/LIGHTGBM/RF)")
    print(f" Mode: {'Full Benchmark' if args.full else 'Fast Demo'} | Budget: {budget} | Pop: {pop_size}")
    print("="*60)
    
    # 1. Get the hyperparameter boundaries
    space = get_xgboost_search_space()
    
    # 2. Initialize Evaluator (Using Pathological XGBoost for edge-case demonstration)
    from automl_engine.core.hard_evaluator import PathologicalXGBoostEvaluator
    evaluator = PathologicalXGBoostEvaluator()
    
    # 3. Run the AutoML Optimizer Engine
    optimizer = DynamicOptimizer(space, evaluator, budget=budget, pop_size=pop_size,
                                 t1_iters=t1_iters, k_iters=k_iters)
    optimizer.run()

if __name__ == "__main__":
    main()

