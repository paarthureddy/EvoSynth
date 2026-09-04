import numpy as np
from automl_engine.core.search_space import get_xgboost_search_space
from automl_engine.core.evaluator import XGBoostEvaluator
from automl_engine.engine.controller import DynamicOptimizer
from ConfigSpace import Configuration


def main():
    print("="*60)
    print(" DYNAMIC EVOLUTIONARY OPTIMIZER - XGBOOST (BREAST CANCER)")
    print("="*60)

    # 1. Get the hyperparameter boundaries
    space = get_xgboost_search_space()

    # 2. Initialize the Evaluator
    print("Loading dataset and initializing XGBoost Evaluator...")
    evaluator = XGBoostEvaluator()

    # 3. Run the AutoML Optimizer Engine
    # Budget: 2000 | Warm-up: 100 | T1: 30 iters | T2: remaining budget
    optimizer = DynamicOptimizer(space, evaluator, budget=2000, pop_size=100,
                                 t1_iters=30, k_iters=10)
    optimizer.run()


if __name__ == "__main__":
    main()
