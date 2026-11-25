"""
CNN Neural Architecture Search demo. Command: `evosynth nas`.
Runs the Dynamic Multi-Armed Bandit optimizer to find the CNN topology and training
hyperparameters for image data.

Settings: [experiments.nas] (budget, pop_size, t1_iters, k_iters, epochs, batch_size, ...),
[search_space.nas].

Output: results/nas/digits/<variant>/seed_<k>/ (dataset becomes fashion_mnist in Phase 3).
"""
from evosynth.config import get_settings
from evosynth.experiments.run_context import RunContext


def run_nas(variant="default", results_dir=None):
    with RunContext("nas", "digits", variant, results_dir=results_dir) as run:
        return _run_nas(run)


def _run_nas(run):
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
                                 pop_size=nas.pop_size, t1_iters=nas.t1_iters, k_iters=nas.k_iters,
                                 population_csv=run.file("population.csv"))
    best = optimizer.run()
    from evosynth.core.decoding import decode_vector_to_dict
    run.write_result({"metric": "validation_accuracy", "best_score": float(best.fitness),
                      "best_config": decode_vector_to_dict(best.vector, space),
                      "winning_algorithm": best.source_algorithm, "evals_used": optimizer.evals_used})
    return best
