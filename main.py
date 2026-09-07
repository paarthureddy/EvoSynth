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
    
    # 2. Initialize Evaluator (Using Pathological XGBoost for edge-case demonstration)
    from automl_engine.core.hard_evaluator import PathologicalXGBoostEvaluator
    evaluator = PathologicalXGBoostEvaluator()
    
    # 3. Run the AutoML Optimizer Engine
    # Budget: 2500 | Warm-up: 100 | T1: 10 iters | T2: remaining budget
    optimizer = DynamicOptimizer(space, evaluator, budget=2500, pop_size=100,
                                t1_iters=10, k_iters=5)
    optimizer.run()

if __name__ == "__main__":
    main()
