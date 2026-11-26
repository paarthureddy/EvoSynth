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
    swap_margin=0.10,
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
| `evaluator` | object | *Required* | Object with an `.evaluate(individual)` method returning a score in % (higher is better). |
| `budget` | `int` | `2000` | Total number of candidate model evaluations before stopping. |
| `pop_size` | `int` | `100` | Initial population N. 40% goes to Tier 1 (four islands, at least 4 members each, so N ≥ 40), 60% to the Tier 2 reserve. The command presets use 40. |
| `t1_iters` | `int` | `30` | Number of iterations for the Tier 1 exploration tournament. |
| `k_iters` | `int` | `10` | Checkpoint interval in Tier 2 iterations. With `engine.auto_k` it is an upper bound, lowered so Tier 2 reaches at least 2 checkpoints. |
| `swap_margin` | `float` | `0.10` | A Tier 1 family replaces the weakest Tier 2 family when its UCB score is higher by more than this (absolute) margin. |
| `stagnation_limit` | `int` | `3` | Consecutive checkpoints without improvement before a Tier 2 island gets random injection. |
| `elite_pct` | `float` | `0.20` | Fraction of top solutions shared between Tier 2 islands. |
| `inject_pct` | `float` | `0.20` | Fraction of stagnant population replaced by random vectors. |
| `max_workers` | `int` or `None` | `None` | Max CPU threads for asynchronous parallel evaluations (`run.max_workers = 0` means `min(8, os.cpu_count())`). |
| `population_csv` | `str` or `None` | `None` | If given, one CSV row per evaluation is written there (the commands pass the run folder's `population.csv`). |

---

## Methods

### `run() -> Individual`
Executes the full optimization lifecycle:
1. Generates and evaluates the initial population in parallel.
2. Splits it into four Tier 1 islands and two reserve groups (stratified by fitness rank, `engine.tier_split`).
3. Runs the Tier 1 competition across GA, PSO, DE and CMA-ES.
4. Scores the islands (composite + UCB1) and promotes the top 2 families to Tier 2.
5. Runs the Tier 2 loop until the budget is used; at every checkpoint: rescoring, SWAP with grace period and rollback, random injection on stagnation, pairwise elite sharing.
6. Returns the best `Individual` found.

After `run()`, `optimizer.evaluated` holds every evaluated individual (used by confirm-before-select), `optimizer.scoring_history` every scoring call with each island's metrics, composite and UCB score, and `optimizer.decisions` every promotion, SWAP, rollback, injection and elite exchange.
