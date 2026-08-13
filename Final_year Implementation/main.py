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
    # Budget: 700 | Warm-up: 50 | T1: 20 iters | T2: remaining budget
    optimizer = DynamicOptimizer(space, evaluator, budget=700, pop_size=50,
                                t1_iters=20, k_iters=5)
    optimizer.run()

if __name__ == "__main__":
    main()
