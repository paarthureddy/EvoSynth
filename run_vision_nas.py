"""
EvoSynth Vision Neural Architecture Search (NAS) Demo Script
Runs the Dynamic Multi-Armed Bandit optimizer to discover the optimal
PyTorch Convolutional Neural Network topology and hyperparameters for image data.
"""
from automl_engine.core.search_space import get_image_nas_search_space
from automl_engine.core.image_evaluator import ImageCNNEvaluator
from automl_engine.engine.controller import DynamicOptimizer

def main():
    print("=" * 60)
    print(" EVOSYNTH: DEEP LEARNING NEURAL ARCHITECTURE SEARCH (NAS)")
    print(" Dataset: Vision / Handwritten Digits (PyTorch CNN)")
    print("=" * 60)

    # 1. Image NAS Search Space (Conv layers, channels, activations, LR, optimizers)
    space = get_image_nas_search_space()

    # 2. PyTorch Vision Evaluator
    evaluator = ImageCNNEvaluator(space=space, epochs=2, batch_size=64)

    # 3. Dynamic Multi-Armed Bandit Optimizer
    optimizer = DynamicOptimizer(
        search_space=space,
        evaluator=evaluator,
        budget=80,
        pop_size=20,
        t1_iters=3,
        k_iters=2
    )

    best_individual = optimizer.run()
    return best_individual

if __name__ == "__main__":
    main()
