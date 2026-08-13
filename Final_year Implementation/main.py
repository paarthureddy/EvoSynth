import numpy as np
from automl_engine.core.search_space import get_cnn_search_space
from automl_engine.core.evaluator import CNNDummyEvaluator
from automl_engine.engine.controller import DynamicOptimizer


def main():
    print("=" * 60)
    print(" EVOSYNTH — HYBRID AUTOML STRATEGY SWITCHING")
    print(" CNN Hyperparameter Search (CIFAR-10 Simulation)")
    print("=" * 60)

    space     = get_cnn_search_space()
    evaluator = CNNDummyEvaluator()

    optimizer = DynamicOptimizer(
        space, evaluator,
        budget=300,
        k_iters=5,
        swap_threshold=0.05,
        stagnation_limit=2,
        elite_pct=0.20,
        inject_pct=0.20,
    )
    optimizer.run()


if __name__ == "__main__":
    main()
