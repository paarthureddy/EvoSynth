
import sys
sys.path.insert(0, '/Users/shivam_kharat/Desktop/RECKON/EvoSynth/Final_year Implementation')
from automl_engine.engine.controller import DynamicOptimizer

from automl_engine.core.search_space import get_cnn_search_space
from automl_engine.core.evaluator import CNNDummyEvaluator
space     = get_cnn_search_space()
evaluator = CNNDummyEvaluator()


optimizer = DynamicOptimizer(
    space, evaluator,
    budget=300,
    k_iters=5,
    step_delay=0.075,
    swap_threshold=0.05,
    stagnation_limit=2,
    elite_pct=0.2,
    inject_pct=0.2,
)
optimizer.run()
