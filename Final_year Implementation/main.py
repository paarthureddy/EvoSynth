import numpy as np
from automl_engine.core.search_space import get_dummy_search_space
from automl_engine.core.evaluator import DummyObjectiveFunction
from automl_engine.engine.controller import DynamicOptimizer
from ConfigSpace import Configuration

def main():
    print("="*60)
    print(" DYNAMIC EVOLUTIONARY OPTIMIZER - DUMMY BENCHMARK")
    print("="*60)
    
    # 1. Get the hyperparameter boundaries
    space = get_dummy_search_space()
    
    # 2. Initialize the Dummy Mathematical Function (Simulates our ML Model)
    evaluator = DummyObjectiveFunction()
    
    # 3. Run the AutoML Optimizer Engine
    # We give it a budget of 300 evaluations (the "outer loop")
    optimizer = DynamicOptimizer(space, evaluator, budget=300, k_iters=5)
    best_ind = optimizer.run()
    
    # 4. Results
    print("\n" + "="*40)
    print("               FINAL RESULTS")
    print("="*40)
    print(f"Best Fitness Score: {best_ind.fitness:.2f}/100")
    print(f"Found by Algorithm: {best_ind.source_algorithm}")
    
    # Decode the mathematical float vector back into real hyperparameter values
    valid_vec = np.copy(best_ind.vector)
    import ConfigSpace.hyperparameters as CSH
    for i, hp_name in enumerate(list(space.keys())):
        hp = space.get_hyperparameter(hp_name)
        if isinstance(hp, CSH.CategoricalHyperparameter):
            num_choices = len(hp.choices)
            idx = int(np.floor(valid_vec[i] * num_choices))
            idx = min(idx, num_choices - 1)
            valid_vec[i] = float(idx)
            
    config = Configuration(space, vector=valid_vec)
    
    print("\n[Best Hyperparameters Discovered]")
    for param in space.values():
        val = config.get(param.name)
        # Format floats nicely for display
        if isinstance(val, float):
            print(f" - {param.name}: {val:.4f}")
        else:
            print(f" - {param.name}: {val}")
        
    print("\n[Secret 'Perfect' Target Values] (For Comparison)")
    print(f" - learning_rate: {evaluator.perfect_lr}")
    print(f" - num_layers: {evaluator.perfect_layers}")
    print(f" - optimizer: {evaluator.perfect_opt}")
    print(f" - dropout: {evaluator.perfect_dropout}")

if __name__ == "__main__":
    main()
