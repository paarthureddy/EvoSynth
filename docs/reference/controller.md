# DynamicOptimizer API Reference

## Class: `DynamicOptimizer`

The primary engine class coordinating the multi-algorithm optimization loop.

```python
from automl_engine.engine.controller import DynamicOptimizer

optimizer = DynamicOptimizer(
    search_space,
    evaluator,
    budget=2000,
    pop_size=100,
    t1_iters=30,
    k_iters=10,
    swap_threshold=0.10,
    stagnation_limit=3,
    elite_pct=0.20,
    inject_pct=0.20,
    max_workers=None
)
```

---

## Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `search_space` | `ConfigSpace.ConfigurationSpace` | *Required* | The hyperparameter boundaries and distributions. |
| `evaluator` | `BaseEvaluator` | *Required* | Object with an `.evaluate(individual)` method returning accuracy. |
| `budget` | `int` | `2000` | Total number of candidate model evaluations before stopping. |
| `pop_size` | `int` | `100` | Size of the total population across Tier 1 and Tier 2. |
| `t1_iters` | `int` | `30` | Number of iterations for the Tier 1 exploration tournament. |
| `k_iters` | `int` | `10` | Checkpoint frequency for composite scoring and elite migration. |
| `stagnation_limit` | `int` | `3` | Consecutive intervals of zero improvement before random injection. |
| `elite_pct` | `float` | `0.20` | Fraction of top solutions shared between Tier 2 islands. |
| `inject_pct` | `float` | `0.20` | Fraction of stagnant population replaced by random vectors. |
| `max_workers` | `int` or `None` | `None` | Max CPU threads for asynchronous parallel evaluations (defaults to `min(8, os.cpu_count())`). |

---

## Methods

### `run() -> Individual`
Executes the full optimization lifecycle:
1. Generates and evaluates the initial population in parallel.
2. Runs the Tier 1 exploration competition across GA, PSO, DE, and CMA-ES.
3. Computes UCB1 composite rankings and promotes the top 2 algorithms to Tier 2.
4. Executes the Tier 2 exploitation loop with elite migration and stagnation safeguards until budget depletion.
5. Returns the best `Individual` found.
