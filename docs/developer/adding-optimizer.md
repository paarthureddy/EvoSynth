# Adding a New Optimizer

EvoSynth's modular design makes integrating new nature-inspired or metaheuristic optimizers straightforward.

---

## The Ask / Tell Contract

Every worker optimizer must inherit or implement the **Ask / Tell** interface:

```python
import numpy as np

class CustomOptimizer:
    def __init__(self, search_space, pop_size):
        self.space = search_space
        self.pop_size = pop_size
        self.population = []

    def ask(self, n: int) -> list[np.ndarray]:
        """
        Generate and return `n` candidate vectors in [0, 1]^d.
        """
        return [np.random.rand(len(self.space)) for _ in range(n)]

    def tell(self, individuals: list) -> None:
        """
        Ingest evaluated individuals (with ind.fitness populated)
        to update internal algorithm state (e.g., velocity, covariance, population).
        """
        self.population = individuals
```

---

## Registering in the Controller

1. Save your optimizer in `src/evosynth/optimizers/custom_opt.py` (put its constants in `settings/optimizers.toml`).
2. In [`src/evosynth/engine/controller.py`](https://github.com/paarthureddy/EvoSynth/blob/main/src/evosynth/engine/controller.py):
   ```python
   from evosynth.optimizers.custom_opt import CustomOptimizer

   def _make_optimizer(algo_key, space, pop_size):
       if algo_key == 'GA': return GeneticAlgorithm(space, pop_size)
       if algo_key == 'PSO': return ParticleSwarm(space, pop_size)
       if algo_key == 'DE': return DifferentialEvolution(space, pop_size)
       if algo_key == 'CMAES': return CMAESOptimizer(space, pop_size)
       if algo_key == 'CUSTOM': return CustomOptimizer(space, pop_size)
       raise ValueError(f"Unknown algo: {algo_key}")
   ```
3. Add `'CUSTOM'` to `self.algo_evals` and `self.cooldowns` in `DynamicOptimizer.__init__`.
