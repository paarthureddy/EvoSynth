# Adding a New Optimizer

EvoSynth's modular design makes integrating new nature-inspired or metaheuristic optimizers straightforward.

---

## The Ask / Tell Contract

Every worker optimizer subclasses `BaseOptimizer` (`src/evosynth/optimizers/base.py`) and implements the **Ask / Tell** interface. `BaseOptimizer` gives it `self.dimensions`, `self.population` and its own seeded random generator `self.rng`; use `self.rng` for all randomness (never `np.random`), so runs stay reproducible from `run.seed`.

```python
from evosynth.optimizers.base import BaseOptimizer


class CustomOptimizer(BaseOptimizer):
    def ask(self, n):
        """Return `n` candidate vectors in [0, 1]^d."""
        return [self.rng.random(self.dimensions) for _ in range(n)]

    def tell(self, individuals):
        """Ingest evaluated individuals (ind.fitness set) and update the optimizer's state."""
        self.population = sorted(self.population + list(individuals),
                                 key=lambda i: i.fitness, reverse=True)[:self.pop_size]

    # Optional: BaseOptimizer.inject(individuals) replaces the worst members (random injection,
    # elite sharing). Override it if the optimizer keeps internal search state, as PSO (particles)
    # and CMA-ES (its distribution) do.
```

---

## Registering in the Controller

1. Save your optimizer in `src/evosynth/optimizers/custom_opt.py` (put its constants in `settings/optimizers.toml`).
2. In [`src/evosynth/engine/controller.py`](https://github.com/paarthureddy/EvoSynth/blob/main/src/evosynth/engine/controller.py):
   ```python
   from evosynth.optimizers.custom_opt import CustomOptimizer

   def _make_optimizer(algo_key, space, pop_size, rng=None):
       if algo_key == 'GA': return GeneticAlgorithm(space, pop_size, rng=rng)
       ...
       if algo_key == 'CUSTOM': return CustomOptimizer(space, pop_size, rng=rng)
       raise ValueError(f"Unknown algo: {algo_key}")
   ```
3. Add `'CUSTOM'` to `self.algo_evals`, `self.cooldowns` and `self._algo_stats` in `DynamicOptimizer.__init__`. Tier 1 currently runs exactly four islands (one per family), so a fifth family also needs the island count in the tier split (`TierManager(n_islands=...)`) and the Tier 1 share to be raised accordingly.
