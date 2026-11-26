# Building a Custom Evaluator

Custom evaluators enable EvoSynth to optimize arbitrary machine learning tasks, custom loss functions, or external simulators.

---

## Evaluator Contract

An evaluator requires only a `.space` attribute and an `.evaluate(individual)` method:

```python
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from evosynth.core.decoding import decode_vector_to_dict

class CustomModelEvaluator:
    def __init__(self):
        # 1. Define hyperparameter boundaries
        self.space = CS.ConfigurationSpace()
        lr = CSH.UniformFloatHyperparameter("learning_rate", lower=1e-4, upper=1e-1, log=True)
        self.space.add_hyperparameter(lr)

        # 2. Load dataset
        self.X, self.y = ...

    def evaluate(self, individual) -> float:
        """
        Receives an Individual containing a [0, 1]^d continuous vector.
        Decodes the vector, trains your model, and returns a fitness score
        (higher is better, e.g. 0 - 100%).
        """
        # Decode the [0, 1] vector into real hyperparameter values (handles log scales,
        # integers and categorical bins the same way as EvoSynth's own evaluators)
        params = decode_vector_to_dict(individual.vector, self.space)

        learning_rate = params["learning_rate"]

        # Train and validate your model
        accuracy = train_and_eval(learning_rate)

        return float(accuracy)
```

---

## Running with the Optimizer

Pass your custom evaluator into `DynamicOptimizer`:

```python
from evosynth.engine.controller import DynamicOptimizer

evaluator = CustomModelEvaluator()
optimizer = DynamicOptimizer(
    search_space=evaluator.space,
    evaluator=evaluator,
    budget=500
)
best_solution = optimizer.run()
```
