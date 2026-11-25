# DynamicOptimizer API Reference

## Class: `DynamicOptimizer`

The primary engine class coordinating the multi-algorithm optimization loop.

```python
from evosynth.engine.controller import DynamicOptimizer

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
    max_workers=None,
    population_csv=None,
)
```

---

## Parameters

Every argument left as `None` takes its value from the active settings
(`[engine]`, `[safeguards]`, `[run]` in [`settings/`](https://github.com/paarthureddy/EvoSynth/tree/main/settings)); the defaults below are those settings.

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
| `max_workers` | `int` or `None` | `None` | Max CPU threads for asynchronous parallel evaluations (`run.max_workers = 0` means `min(8, os.cpu_count())`). |
| `population_csv` | `str` or `None` | `None` | If given, one CSV row per evaluation is written there (the commands pass the run folder's `population.csv`). |

---

## Methods

### `run() -> Individual`
Executes the full optimization lifecycle:
1. Generates and evaluates the initial population in parallel.
2. Runs the Tier 1 exploration competition across GA, PSO, DE, and CMA-ES.
3. Computes UCB1 composite rankings and promotes the top 2 algorithms to Tier 2.
4. Executes the Tier 2 exploitation loop with elite migration and stagnation safeguards until budget depletion.
5. Returns the best `Individual` found.
