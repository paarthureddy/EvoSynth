"""
CNN Neural Architecture Search demo. Command: `evosynth nas`.
Runs the Dynamic Multi-Armed Bandit optimizer to find the CNN topology and training
hyperparameters for image data.

Settings: [experiments.nas] (budget, pop_size, t1_iters, k_iters, epochs, batch_size, ...),
[search_space.nas].
"""
from evosynth.config import get_settings


def run_nas():
    from evosynth.engine.controller import DynamicOptimizer
    from evosynth.evaluators.image import ImageCNNEvaluator
    from evosynth.search_spaces.nas import get_image_nas_search_space

    nas = get_settings().experiments.nas
    print("=" * 60)
    print(" EVOSYNTH: DEEP LEARNING NEURAL ARCHITECTURE SEARCH (NAS)")
    print(" Dataset: Vision / Handwritten Digits (PyTorch CNN)")
    print("=" * 60)

    # 1. Image NAS search space (channels, units, activation, dropout, LR, optimizer)
    space = get_image_nas_search_space()
    # 2. PyTorch vision evaluator (epochs / batch size from [experiments.nas])
    evaluator = ImageCNNEvaluator(space=space)
    # 3. Dynamic Multi-Armed Bandit optimizer
    optimizer = DynamicOptimizer(search_space=space, evaluator=evaluator, budget=nas.budget,
                                 pop_size=nas.pop_size, t1_iters=nas.t1_iters, k_iters=nas.k_iters)
    return optimizer.run()
