# 03 — Engine Internals (`DynamicOptimizer`)

This file walks through `src/evosynth/engine/controller.py` and the modules it uses (`scoring.py`, `bandit.py`, `tiers.py`) exactly as they are implemented after the fix plan (Phases 0–4). Every constant comes from `settings/engine.toml`; the values quoted are the defaults.

---

## 1. State held by the controller

| Attribute | Meaning |
|---|---|
| `evals_used` | Evaluations consumed so far, across all optimizers |
| `algo_evals[algo]` | Evaluations consumed by each optimizer family (GA/PSO/DE/CMAES), counted across both tiers. Initial-population evaluations count for no family. |
| `best_individual` | Global best Individual |
| `history` | One entry per evaluation: `{eval_id, fitness, best_so_far, algo}` |
| `evaluated` | Every evaluated Individual (used by confirm-before-select after the run) |
| `_algo_stats[algo]` | `best`, `prev_best` (best at the family's last scoring call), `eval_seconds` (evaluation time since then), `best_history` (best at every scoring call) |
| `scoring_history` | One entry per scoring call: evaluations used and every island's fitness, speed, efficiency, diversity, trend, composite and UCB score |
| `decisions` | Every promotion, SWAP, rollback, injection and elite exchange, with the evaluation count |
| `cooldowns[algo]` | Iterations remaining before this family may be swapped into Tier 2 |
| `active_grace_period` | `None`, or `{algo, iters_left, baseline, snapshot, target_t1}` |
| `composite_engine` | `CompositeScoringEngine()` |
| `bandit` | `UCB1Bandit()` with `c = bandit.c` (0.1); makes every ranking decision |
| `rng` | The controller's own generator, `stream("engine")`, derived from `run.seed` |

Every optimizer the controller creates gets its own generator, `stream("engine", "optimizer", algo, n)`, so a run is fully reproducible from `run.seed`, whatever the thread scheduling (Phase 0).

---

## 2. Step-by-step

### STEP 0 — Search space

The controller prints the dimension count `d = len(space.keys())`, the population size, the budget, the number of T1 iterations and the tier-split method. It emits `step {num:0}`.

### STEP 1 — Initial population

- Random vectors `self.rng.random(d)` are evaluated in chunks of `max(1, 2·max_workers)`, tagged `'Init'`.
- A progress line and a `progress` event are produced after each chunk.

### STEP 2 — Tier split (`TierManager`, `engine/tiers.py`)

```
island_size = int(pop_size · t1_fraction) // 4        # t1_fraction = 0.40
Tier 1      = 4 · island_size individuals             # four islands
reserve     = the rest, dealt into 2 groups
```

With the default **stratified** split (`engine.tier_split`):

1. The population is sorted by fitness and cut into `4 · island_size` equal rank strata; one member is drawn at random from each stratum for Tier 1. Tier 1 therefore spans the whole fitness range instead of being "the best 40%".
2. Those members are dealt to the four islands block by block (one rank block per round, each member to a random island), so every island gets a similar mix of good and weak members and no family starts with an advantage.
3. The reserve is dealt into two groups the same way.

The split uses its own generator (`stream("engine", "tiers")`). The alternatives, kept for the split ablation, are `random` (plain shuffle) and `elite` (the v1 behaviour: sorted top 40% in contiguous slices, GA getting the very best).

Islands must have at least `engine.min_island_size` = 4 members, so `pop_size` must be at least 40; smaller values are rejected when the controller is created.

| pop_size N | Tier 1 total | Per island | Reserve | Each reserve group | Each T2 island after promotion |
|---|---|---|---|---|---|
| 40 (presets) | 16 | 4 | 24 | 12 | 4 + 12 = 16 |
| 60 | 24 | 6 | 36 | 18 | 6 + 18 = 24 |
| 100 (`[engine]` default) | 40 | 10 | 60 | 30 | 10 + 30 = 40 |

**Checkpoint interval k (auto_k).** Right after the split, `_set_checkpoint_interval` estimates how many Tier 2 iterations the budget allows:

```
remaining = budget − pop_size − t1_iters · 4 · island_size
t2_iters  = remaining // pop_size                      # one T2 iteration costs about N evaluations
k         = max(1, min(k_iters, t2_iters // min_t2_checkpoints))   # min_t2_checkpoints = 2
```

If even `k = 1` cannot give two checkpoints, the run prints a warning. Example: `evosynth run --full` (budget 500, N 40, t1_iters 5) → remaining 380, about 9 Tier 2 iterations, k = min(3, 4) = 3.

### STEP 3 — Tier 1 tournament (`t1_iters` iterations)

1. One optimizer per family is created with `pop_size = island_size` and told its island.
2. Each iteration, for every family: `inds = _evaluate_batch(opt.ask(island_size), "T1_<algo>")`, then `opt.tell(inds)`. Each iteration costs `4 × island_size` evaluations.
3. A row is printed with each family's island best, the global best, the time taken and budget use; `<-- NEW BEST` marks a new global best.
4. **T1 stagnation check.** On iterations where `i % k == 0`, the four islands are scored, and any island whose best did not improve since its last scoring call gets **50%** of its members replaced by fresh random vectors (`_random_inject`, §5).

### STEP 4 — Composite scoring and top-2 promotion

1. `_compute_composite` scores the four T1 islands (§3); every score is also recorded in `bandit` and `scoring_history`. The table (Algorithm, Best Acc, Diversity, Composite, Bonus, UCB Score) is printed.
2. `bandit.rank(scores)` orders the families by UCB score: the top `n_promoted` = 2 are **promoted to Tier 2**, the other two **stay in Tier 1**.
3. For promoted family `idx ∈ {0, 1}`, a new optimizer of the same family is created with its Tier 1 island **plus reserve group `idx`**; its size is the sum (16 with N = 40).
4. The two other families keep their Tier 1 optimizers.
5. `tier_state` and `switch{action:"Promotion"}` are emitted and recorded.

### STEP 5 — Tier 2 working loop (until `evals_used ≥ budget`)

Each iteration:

1. Decrement every positive cooldown.
2. Every T2 optimizer runs `ask(opt.pop_size)` → evaluate (tag `T2_<algo>`) → `tell`; every T1 optimizer does the same (tag `T1_<algo>`).
3. Print the row and emit `progress`.
4. **Grace-period bookkeeping**, if a grace period is active:
   - `iters_left -= 1`.
   - If the global best is now above `baseline`, the grace period ends successfully.
   - Otherwise, if `iters_left ≤ 0`, a **ROLLBACK** happens: the demoted optimizer's snapshot is restored into Tier 2, the challenger is rebuilt in Tier 1 with `island_size` fresh random (evaluated) individuals, the demoted family leaves Tier 1, and the challenger gets a cooldown of `2k`.
5. **Checkpoint**, when `iteration % k == 0`. All T2 and T1 islands are scored and the table is printed. Then, in order:

   **a) SWAP.** With `best_t1` the T1 family ranked first and `worst_t2` the T2 family ranked last by the bandit, a swap happens when

   ```
   UCB(best_t1) > UCB(worst_t2) + swap_margin        (additive, 0.10)
   and cooldowns[best_t1] == 0
   and no grace period is active
   ```

   1. A grace period starts: `{algo: best_t1, iters_left: 5, baseline: global best, snapshot: deepcopy(T2 optimizer of worst_t2), target_t1: worst_t2}`.
   2. `cooldowns[worst_t2] = 2k`.
   3. A new T2 optimizer of family `best_t1` (old T2 size) is told the old T1 island plus the tail of the old T2 population.
   4. A new T1 optimizer of family `worst_t2` (island size) is told the **top** `island_size` members of the old T2 population.
   5. `switch{action:"Swap"}` and `tier_state` are emitted and recorded.

   **b) T2 stagnation.** For each T2 island whose best did not improve since its last scoring call, `stagnation_counters[algo] += 1` (otherwise it resets). At `stagnation_limit` = 3, the island gets **20%** random injection and the counter resets.

   **c) Pairwise elite sharing (Tier 2 only).** With `n = max(1, int(min(|A|, |B|) · 0.20))`, island A receives island B's top n and island B receives island A's top n, each through `optimizer.inject` (which replaces the receiver's worst n and updates its search state). Tier 1 islands do not take part.
6. Reprint the header and continue.

### STEP 6 — Final results

The controller prints total evaluations, total time, the winning algorithm tag (the `source_algorithm` of the best individual, e.g. `T2_PSO`), the best score and every decoded hyperparameter, and returns the best `Individual`. The calling command (`evosynth run` / `scale`) then runs confirm-before-select and the refit ([02-architecture.md](02-architecture.md) §3).

---

## 3. Scoring: exact formulas

`_compute_composite(islands)` computes, for each island `P` of family `a` (d = number of dimensions):

```
best        = max fitness in P
improvement = max(best − _algo_stats[a].prev_best, 0)              # "speed"
efficiency  = improvement / max(_algo_stats[a].eval_seconds, 1e-6)  # gain per evaluation-second
diversity   = mean pairwise Euclidean distance between P's latent vectors / sqrt(d)   (0 if |P| < 2)
trend       = least-squares slope of the family's best over its last trend_window = 3 scoring calls
then: prev_best ← best, eval_seconds ← 0, best appended to best_history
```

`eval_seconds` accumulates the wall time of the family's evaluation batches between scoring calls.

`CompositeScoringEngine.calculate_scores` then min-max normalises each metric **across the islands scored together**:

```
norm_m(a)    = (m(a) − min_m) / (max_m − min_m)     if max_m − min_m > 1e-9
             = 0.5  (tie_value)                     otherwise
composite(a) = 0.30·norm_fitness + 0.25·norm_speed + 0.15·norm_efficiency
             + 0.15·norm_diversity + 0.15·norm_trend
```

The bandit then adds the exploration bonus:

```
bonus(a) = c · sqrt( ln(max(1, evals_used)) / max(1, algo_evals[a]) ),   c = 0.1
UCB(a)   = composite(a) + bonus(a)
```

**What follows from these definitions:**

- **Scores are relative.** Because of min-max normalisation, a composite only means something within one scoring call: the best island on a metric gets 1, the worst 0.
- **Efficiency separates cheap progress from expensive progress.** Two families with the same improvement get different efficiency if one used more evaluation time (e.g. larger forests).
- **Diversity is measured on configurations, not scores.** Two islands with identical accuracies but different hyperparameters are told apart.
- **Speed and trend differ.** Speed is the gain since the last call; trend is the slope over the last three calls, so one lucky jump moves speed fully but trend only partly.
- **The bonus is small.** With `c = 0.1` it is typically 0.04–0.08; it mainly breaks near-ties in favour of families that have used less budget.

---

## 4. Worked example: a Tier 1 scoring call

An illustrative Step 4 with N = 40 and `t1_iters = 3` (the `evosynth run` preset): each family has used 3 × 4 = 12 evaluations and the run 40 + 48 = 88. The numbers below were computed with the engine's own `CompositeScoringEngine` and `UCB1Bandit`.

Raw metrics:

| Family | Fitness (%) | Speed | Efficiency | Diversity | Trend |
|---|---|---|---|---|---|
| GA | 86.2 | 0.4 | 0.04 | 0.31 | 0.20 |
| PSO | 85.1 | 1.1 | 0.12 | 0.42 | 0.55 |
| DE | 84.6 | 0.0 | 0.00 | 0.47 | 0.00 |
| CMAES | 85.6 | 0.6 | 0.05 | 0.22 | 0.30 |

For GA: fitness (86.2 − 84.6)/1.6 = 1 → 0.300; speed 0.4/1.1 = 0.364 → 0.091; efficiency 0.04/0.12 = 0.333 → 0.050; diversity (0.31 − 0.22)/0.25 = 0.36 → 0.054; trend 0.20/0.55 = 0.364 → 0.055; composite **0.549**. The bonus is `0.1 · sqrt(ln 88 / 12) = 0.061` for every family (equal use so far).

| Family | Composite | Bonus | UCB | Decision |
|---|---|---|---|---|
| PSO | 0.764 | 0.061 | **0.825** | promoted |
| GA | 0.549 | 0.061 | **0.611** | promoted |
| CMAES | 0.468 | 0.061 | 0.529 | stays in Tier 1 |
| DE | 0.150 | 0.061 | 0.211 | stays in Tier 1 |

PSO is promoted ahead of GA despite a lower best score, because it is improving fastest and most cheaply. At a later checkpoint, CMAES could SWAP in only if its UCB exceeded the weaker Tier 2 family's by more than 0.10.

Real scoring tables from the re-runs are in each run's `run.log`, and every value is stored in the controller's `scoring_history` (the ablation's `result.json` reports them).

---

## 5. Random injection (`_random_inject`)

```
n_inject = max(1, int(len(opt.population) · pct))
fresh    = _evaluate_batch([rng.random(d) for _ in range(n_inject)], tag)   # costs budget
opt.inject(fresh)
```

`inject` replaces the optimizer's worst members and keeps its internal state consistent: PSO resets the replaced particles (and its global best if a newcomer beats it), CMA-ES restarts its distribution around the new population, DE clears its pending targets.

| Where | Fraction | Trigger |
|---|---|---|
| Tier 1 tournament (Step 3) | 50% (`t1_inject_pct`) | No improvement at a T1 check |
| Tier 2 (Step 5) | 20% (`t2_inject_pct`) | No improvement at 3 consecutive checkpoints |

---

## 6. Budget accounting

- The budget is counted in **evaluations**. It is shared, not multiplied, so four optimizers do not cost four times as much.
- `_evaluate_batch` truncates any request to the remaining budget, so the final batch can be partial. Step 5 exits as soon as `evals_used ≥ budget`.
- Injections and rollback re-seeding consume budget. Confirm-before-select (after the run) does not; its cost is reported separately in `result.json`.
- The meta-logic itself (scoring, sorting, deep copies) costs negligible time compared with model training.

---

## 7. Helper engine classes

### `CompositeScoringEngine` (`engine/scoring.py`)

Holds the weights `{fitness 0.30, speed 0.25, efficiency 0.15, diversity 0.15, trend 0.15}` and `tie_value` 0.5 from `[composite]`. `calculate_scores(dict)` returns the normalised weighted sums of §3.

### `UCB1Bandit` (`engine/bandit.py`)

- `c` defaults to `bandit.c` (0.1).
- `update(algo, reward)` records a pull with the composite as reward (called at every scoring call); `mean_reward(algo)`.
- `bonus(n_algo, n_total)` and `score(algo, composite, n_algo, n_total) = composite + bonus`.
- `rank(ucb_scores)` orders families best first; the controller uses it for promotion and both sides of the SWAP rule.

### `TierManager` (`engine/tiers.py`)

- `TierManager(population, island_size, n_islands=4, split="stratified", rng)` performs the split of Step 2; `islands()` and `reserve_groups(n)` deal the members.
- `select_elites(pop, top_pct)` and `share_elites(a, b, top_pct)` (pairwise exchange) are used for elite sharing.

---

## 8. What changed from the panel-review version (v1)

| Topic | v1 (panel review) | Now |
|---|---|---|
| Who decides | An inline UCB-style formula; `UCB1Bandit` only did bookkeeping (default c 1.414) | `UCB1Bandit` (c 0.1) ranks every promotion and SWAP |
| Efficiency | Constant 1.0 (no effect) | Improvement per evaluation-second |
| Trend | Identical to speed | Slope over the last 3 scoring calls |
| Diversity | Spread of fitness values (always 0.5 for 2-member islands) | Mean latent distance / √d; islands of at least 4 |
| Tier split | Sorted top 40% in contiguous slices; GA got the best | Stratified by rank; `random` / `elite` for the ablation |
| Checkpoint k | Fixed; Tier 2 sometimes never reached a checkpoint | Sized automatically (≥ 2 Tier 2 checkpoints) |
| Injection | Overwrote the population list only | `optimizer.inject` updates the search state |
| Elite sharing | Pooled: both islands got the same elites | Pairwise exchange |
| Stagnation | Zero "trend" (= zero improvement) | Zero improvement since the last scoring call |
| Randomness | Global NumPy state; CMA-ES reseeded from the clock | One seed; a generator per component |
| CMA-ES | Updates often failed silently (seeds told with the first generation) | One update per generation with its own samples |
| PSO / DE | PSO re-evaluated its seeds; both only used the first members when asked for fewer | Seeds are moved first; members visited round-robin; DE one-to-one selection |
| Records | Printed tables only | `scoring_history`, `decisions`, `evaluated` |

---

## Quiz

**1.** Which object makes the promotion and SWAP decisions, and with what exploration constant?
<details><summary>Answer</summary>`UCB1Bandit`, with c = 0.1 (`bandit.c`).</details>

**2.** Write the five composite metrics and their weights.
<details><summary>Answer</summary>Fitness 0.30, speed 0.25, efficiency 0.15, diversity 0.15, trend 0.15.</details>

**3.** How is diversity computed, and why divide by √d?
<details><summary>Answer</summary>Mean pairwise Euclidean distance between the island's latent vectors, divided by √d (the largest possible distance in [0,1]^d), so it lies between 0 and 1.</details>

**4.** A family improved its best by 0.6 points using 3 seconds of evaluation time since its last scoring call. What are its speed and efficiency?
<details><summary>Answer</summary>Speed 0.6; efficiency 0.6 / 3 = 0.2 per second.</details>

**5.** Why is a composite score only meaningful within one scoring table?
<details><summary>Answer</summary>Each metric is min-max normalised across the islands scored together, so scores are relative to the other islands at that moment.</details>

**6.** With N = 60, what are the Tier 1 island size and the size of each Tier 2 island after promotion?
<details><summary>Answer</summary>Island size int(60·0.4)//4 = 6; reserve 36 in two groups of 18; each Tier 2 island 6 + 18 = 24.</details>

**7.** Budget 500, N 40, t1_iters 5, k_iters 3. What k does auto_k choose?
<details><summary>Answer</summary>remaining = 500 − 40 − 5·16 = 380; t2_iters = 380 // 40 = 9; k = max(1, min(3, 9 // 2)) = 3.</details>

**8.** State the SWAP rule exactly.
<details><summary>Answer</summary>Swap when UCB(best Tier 1) > UCB(worst Tier 2) + 0.10, the Tier 1 family has no cooldown, and no grace period is active.</details>

**9.** What happens if a swapped-in family fails to improve the global best within its grace period?
<details><summary>Answer</summary>Rollback: the demoted family's snapshot returns to Tier 2, the challenger is rebuilt in Tier 1 with fresh random members, and it gets a cooldown of 2k.</details>

**10.** How does pairwise elite sharing differ from the v1 pooled sharing?
<details><summary>Answer</summary>Each Tier 2 island receives the other island's top 20% (in place of its own worst 20%); in v1 both islands received the same pooled elites.</details>

**11.** When does a Tier 2 island get random injection, and how much?
<details><summary>Answer</summary>After 3 consecutive checkpoints without improvement; 20% of its members are replaced by fresh random vectors (through `inject`).</details>

**12.** In the worked example, why is PSO promoted ahead of GA although GA has the higher best score?
<details><summary>Answer</summary>PSO's speed, efficiency, diversity and trend are the highest, which outweighs GA's fitness advantage in the composite (0.764 vs 0.549).</details>

**13.** Do confirm-before-select fits count against the evaluation budget?
<details><summary>Answer</summary>No; they happen after the search and are reported separately in result.json.</details>

**14.** Name three records the controller keeps after `run()` for later analysis.
<details><summary>Answer</summary>`evaluated` (all individuals), `scoring_history` (every scoring call's metrics, composite and UCB) and `decisions` (promotions, swaps, rollbacks, injections, elite exchanges).</details>
