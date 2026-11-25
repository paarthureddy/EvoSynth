# 03 — Engine Internals (`DynamicOptimizer`)

This file walks through `automl_engine/engine/controller.py` and the scoring modules exactly as they are implemented. All constants and formulas are taken from the code. Where the docs, paper or slides describe something differently, §9 lists the difference.

---

## 1. State held by the controller

| Attribute | Meaning |
|---|---|
| `evals_used` | Evaluations consumed so far, across all optimizers |
| `algo_evals[algo]` | Evaluations consumed by each optimizer family (GA/PSO/DE/CMAES), counted across both tiers. Init evaluations are not counted for any family. |
| `best_individual` | Global best Individual |
| `history` | One entry per evaluation: `{eval_id, fitness, best_so_far, algo}` |
| `_algo_stats[algo]` | `best` (best fitness seen by that family) and `prev_best` (best at the last scoring call, used for trend) |
| `cooldowns[algo]` | Iterations remaining before this family may be swapped into Tier 2 |
| `active_grace_period` | `None`, or `{algo, iters_left, baseline, snapshot, target_t1}` |
| `composite_engine` | `CompositeScoringEngine()` |
| `bandit` | `UCB1Bandit()` (see §8) |
| `c` | 0.1, the exploration coefficient used in scoring |

---

## 2. Step-by-step

### STEP 0 — Search space

The controller prints the dimension count `d = len(space.keys())`, the population size, the budget and the number of T1 iterations. It emits `step {num:0}`.

### STEP 1 — Initial population

- Random vectors `np.random.rand(d)` are evaluated in chunks of `max(1, 2·max_workers)`, tagged `'Init'`.
- A progress line and a `progress` event are produced after each chunk.
- The full population is then **sorted by fitness, best first**.

### STEP 2 — Tier split

```
t1_count       = int(pop_size * 0.4)
t1_island_size = t1_count // 4
seeds: GA    = full_pop[0 : s]
       PSO   = full_pop[s : 2s]
       DE    = full_pop[2s : 3s]
       CMAES = full_pop[3s : t1_count]          (s = t1_island_size)
t2_reserved    = full_pop[t1_count:]            # the worse 60%
```

The slices are **contiguous ranks** of the sorted population:
- GA receives the very best individuals, PSO the next block, and so on.
- The Tier 2 reserve is the bottom 60% by fitness.

| pop_size | T1 total | Per island | Reserve | Each T2 island after promotion |
|---|---|---|---|---|
| 100 | 40 | 10 | 60 | 10 + 30 = 40 |
| 50 | 20 | 5 | 30 | 5 + 15 = 20 |
| 30 | 12 | 3 | 18 | 3 + 9 = 12 |
| 20 | 8 | 2 | 12 | 2 + 6 = 8 |

### STEP 3 — Tier 1 tournament (`t1_iters` iterations)

1. One optimizer is created per family with `pop_size = t1_island_size` and told its seed slice.
2. Each iteration, for every family: `inds = _evaluate_batch(opt.ask(t1_island_size), "T1_<algo>")`, then `opt.tell(inds)`. So each iteration costs `4 × t1_island_size` evaluations.
3. A row is printed with each family's island best, the global best, the time taken and budget use. `<-- NEW BEST` is appended when the global best improves.
4. **T1 stagnation check.** On iterations where `i % k == 0`:
   - the composite metrics of the four islands are computed;
   - any island whose `trend == 0.0` gets **50%** of its population replaced through `_random_inject(opt, 0.50, ...)`, which evaluates fresh random vectors and drops the worst.
5. The loop breaks early if the budget runs out.

### STEP 4 — Composite scoring and top-2 promotion

1. `_compute_composite` is run on the four T1 islands, and the table is printed with columns Algorithm, Best Acc, Diversity, Composite, Bonus and UCB Score.
2. Islands are sorted by UCB score:
   - `top2` are **PROMOTED to Tier 2**;
   - `bot2` are **RETAINED in Tier 1**.
3. Each family is registered in `self.bandit` and `bandit.update(algo, score)` is called once.
4. Tier 2 optimizers are built. For winner `idx ∈ {0, 1}`:
   - `t2_size = len(its T1 island) + len(reserve)//2`;
   - it receives a new optimizer of the same family, told its T1 island plus `reserve[idx·half : (idx+1)·half]`.
5. The two losing families keep their existing T1 optimizers.
6. The events `tier_state` and `switch{action:"Promotion"}` are emitted.

### STEP 5 — Tier 2 working loop (until `evals_used ≥ budget`)

Each iteration:

1. Decrement every positive cooldown.
2. Every T2 optimizer runs `ask(opt.pop_size)` → evaluate (tag `T2_<algo>`) → `tell`. Every T1 optimizer does the same (tag `T1_<algo>`).
3. Print the row and emit `progress`.
4. **Grace-period bookkeeping**, if a grace period is active:
   - `iters_left -= 1`.
   - If the global best is now greater than `baseline`, the grace period ends successfully.
   - Otherwise, if `iters_left ≤ 0`, a **ROLLBACK** happens:
     1. The snapshot of the demoted optimizer is restored into Tier 2 under its old key, and the promoted family is removed from Tier 2.
     2. The promoted family is rebuilt in Tier 1 with `t1_island_size` *fresh random* individuals, which are evaluated.
     3. The demoted family is removed from Tier 1.
     4. The failed challenger gets `cooldowns = 2k`.
     5. `switch{action:"Rollback"}` and `tier_state` are emitted.
5. **Checkpoint**, when `iteration % k == 0`. Scores are computed for all T2 and T1 islands and the table is printed. Then three steps run in order:

   **a) SWAP.** Let `best_t1` be the T1 island with the highest UCB score and `worst_t2` the T2 island with the lowest. A swap happens when all three conditions hold:

   ```
   UCB(best_t1) > UCB(worst_t2) + swap_threshold      (additive, threshold 0.10)
   and cooldowns[best_t1] == 0
   and no grace period is active
   ```

   The swap then proceeds as follows:
   1. A grace period starts: `{algo: best_t1, iters_left: 5, baseline: current global best, snapshot: deepcopy(T2 optimizer of worst_t2), target_t1: worst_t2}`.
   2. `cooldowns[worst_t2] = 2k`.
   3. A new T2 optimizer of family `best_t1` is created with the old T2 size. It is told the old T1 island plus the tail of the old T2 population: `old_t1 + old_t2[len(old_t1):]`.
   4. A new T1 optimizer of family `worst_t2` is created with the T1 island size and told the **top** `t1_island_size` individuals of the old T2 population.
   5. Dictionaries and stagnation counters are updated, and `switch{action:"Swap"}` and `tier_state` are emitted.

   **b) T2 stagnation.**
   - For each T2 island whose `trend == 0`, `stagnation_counters[algo] += 1`, otherwise the counter resets to 0.
   - When a counter reaches `stagnation_limit` (3), the island gets `_random_inject(opt, inject_pct=0.20, ...)` and the counter resets.

   **c) Elite sharing (Tier 2 only).**
   1. For each T2 island, take its top `n = max(1, int(pop_size · 0.20))` individuals and add them to `elite_pool`.
   2. Sort the pool, best first.
   3. Each T2 island replaces its worst `n` individuals with the pool's top `n`.

   Both islands receive the *same* best elites. Tier 1 islands never take part.
6. Reprint the header and continue.

### STEP 6 — Final results

The controller prints total evaluations, total time, the winning algorithm tag (the `source_algorithm` of the best individual, e.g. `T2_PSO`), best accuracy, the optimal model type and every decoded hyperparameter. It returns the best `Individual`.

---

## 3. Scoring: exact formulas

`_compute_composite(algo_pops)` computes, for each island `P` (a list of Individuals) of family `a`:

```
f_i       = fitness values in P
best      = max(f_i)
spread    = max(f_i) − min(f_i) + 1e-9
diversity = min( std(f_i) / spread , 1.0 )          # numpy population std
prev      = _algo_stats[a].prev_best                 # family-level, updated every call
trend     = max(best − prev, 0)
_algo_stats[a].prev_best = best
raw metrics = { fitness: best, speed: trend, efficiency: 1.0, diversity: diversity, trend: trend }
```

`CompositeScoringEngine.calculate_scores(raw)` then min–max normalises each metric **across the islands being compared**:

```
norm_m(a) = (m(a) − min_m) / (max_m − min_m)      if max_m − min_m > 1e-9
          = 0.5                                   otherwise
composite(a) = 0.30·norm_fitness + 0.25·norm_speed + 0.15·norm_efficiency
             + 0.15·norm_diversity + 0.15·norm_trend
```

The exploration bonus and UCB score are:

```
bonus(a) = c · sqrt( ln(max(1, evals_used)) / max(1, algo_evals[a]) ),   c = 0.1
UCB(a)   = composite(a) + bonus(a)
```

**What follows from these definitions:**

- **Efficiency is constant.** It is always 1.0, so it normalises to 0.5 for every island and adds a constant 0.075. It never changes a ranking.
- **Speed and trend are the same quantity.** Together they make up 40% of the composite weight and reward recent improvement.
- **Diversity is fitness-based, not location-based.** It measures how spread an island's fitness values are, not distance in the latent space.
- **Diversity is fixed at 0.500 for islands of two.** For two values `a ≠ b`, `std = |a−b|/2`, so `std/spread = 0.5` exactly. With identical values it is 0. This is why 0.500 appears so often in logs with small islands (pop 20 → 2 per T1 island). The UPGS/CVGS ablation summaries also report a *hardcoded* 0.500/0.450 for EvoSynth; see [08-results.md](08-results.md).
- **Trend depends on call history.** `prev_best` is stored per family and updated on *every* scoring call (T1 stagnation checks, promotion and each checkpoint). Trend is therefore "improvement since the last time this family was scored".
- **Scores are relative.** Because of min–max normalisation, a composite score only means something within one table. The best island on a metric gets 1 and the worst gets 0.
- **The bonus is small.** With `c = 0.1` it is typically 0.04–0.08. It mainly breaks near-ties in favour of families that have consumed less budget.

---

## 4. Worked example: reproducing the real Step 4 table

These numbers come from the Covertype 120-evaluation run (`results_forest/Test1_19_Nov/evosynth/run_120.txt`, which is identical to `results/covertype_run_with_fe.txt`). The run used pop = 20, `t1_iters = 3` and `k = 2`, so each T1 island has 2 individuals.

Tier 1 log:

```
Iter   GA      PSO     DE      CMAES   Global  Budget
 1   85.96%  83.98%  83.20%  81.97%  85.96%   28/120
 2   86.21%  83.98%  83.20%  81.97%  86.21%   36/120   ← k=2: scoring call sets prev_best
 3   86.21%  83.98%  83.20%  82.66%  86.21%   44/120
```

The Step 4 raw metrics, with trend measured since iteration 2:

| Family | Best | Diversity | Trend (= speed) |
|---|---|---|---|
| CMAES | 82.66 | 0.5 | 0.69 |
| GA | 86.21 | 0.5 | 0 |
| DE | 83.20 | 0.5 | 0 |
| PSO | 83.98 | 0.0 (both members equal) | 0 |

Normalising (fitness range 82.66–86.21 = 3.55; diversity 0–0.5; trend 0–0.69):

| Family | 0.30·F | 0.25·S | 0.075 (Eff) | 0.15·D | 0.15·T | Composite | Bonus | UCB |
|---|---|---|---|---|---|---|---|---|
| CMAES | 0 | 0.25 | 0.075 | 0.15 | 0.15 | **0.625** | 0.079 | **0.704** |
| GA | 0.30 | 0 | 0.075 | 0.15 | 0 | **0.525** | 0.079 | **0.604** |
| DE | 0.30·0.152 = 0.046 | 0 | 0.075 | 0.15 | 0 | **0.271** | 0.079 | **0.350** |
| PSO | 0.30·0.372 = 0.112 | 0 | 0.075 | 0 | 0 | **0.187** | 0.079 | **0.266** |

The bonus is `0.1·sqrt(ln 44 / 6) = 0.079`: 44 evaluations used, 6 per family (3 iterations × 2).

These values match the log exactly, so **CMAES and GA are promoted**. CMAES is promoted with the *lowest* accuracy because it was the only family still improving. That is the intended behaviour of the trend and speed weighting.

**The live SWAP at the first Tier 2 checkpoint** (iteration 2 of Step 5):

```
T1_PSO   85.20%  div 0.500  comp 0.654  bonus 0.067  UCB 0.720
T2_CMAES 84.71%  div 0.335  comp 0.620  bonus 0.045  UCB 0.665
T2_GA    86.33%  div 0.450  comp 0.503  bonus 0.045  UCB 0.547
T1_DE    83.20%  div 0.500  comp 0.225  bonus 0.067  UCB 0.292
[SWAP] T1_PSO (0.720) beats T2_GA (0.547) by >10%!
```

`0.720 > 0.547 + 0.10`, so PSO moves to Tier 2 and GA moves to Tier 1. The final winner was `T2_PSO` at 86.39% CV. Tier 2 bonuses are smaller because T2 families have consumed more evaluations.

---

## 5. Random injection (`_random_inject`)

```
n_inject = max(1, int(len(opt.population) * pct))
fresh    = _evaluate_batch([rand(d) for _ in range(n_inject)], tag)   # costs budget
opt.population.sort(desc); opt.population = population[:-n_inject] + fresh
```

| Where | Fraction | Trigger |
|---|---|---|
| Tier 1 tournament (Step 3) | 50% | Trend 0 at a T1 checkpoint |
| Tier 2 (Step 5) | 20% | Trend 0 at 3 consecutive checkpoints |

The function edits `opt.population` directly. For PSO and CMA-ES the injected individuals change the tracked population, but not the particles' velocities or the CMA distribution; those adapt only through later `tell` calls.

---

## 6. Budget accounting

- The budget is counted in **evaluations**. It is shared, not multiplied, so four optimizers do not cost four times as much.
- `_evaluate_batch` truncates any request to the remaining budget, so the final batch can be partial. Step 5 exits as soon as `evals_used ≥ budget`.
- Injections, swap-related evaluations and rollback re-seeding all consume budget.
- The meta-logic itself (scoring, sorting, deep copies) costs negligible time compared with model training.

---

## 7. Helper engine classes

### `CompositeScoringEngine` (`composite.py`)

Holds the weights `{fitness:0.30, speed:0.25, efficiency:0.15, diversity:0.15, trend:0.15}`. `calculate_scores(dict)` returns normalised weighted sums as in §3.

### `UCB1Bandit` (`ucb1.py`)

- `__init__(c=1.414)` sets up the exploration constant, a `stats` dict mapping algo → {pulls, total_reward}, and `total_pulls`.
- `register_algorithm`, and `update(algo, reward)` which adds a pull and adds the reward.
- `select_next(available)` returns any algorithm with zero pulls; otherwise it returns the argmax of `avg_reward + c·sqrt(ln(total_pulls)/pulls)`.

### `TierManager` (`tier_manager.py`)

- Shuffles a population and splits it 40/60 (`t1_ratio = 0.4`).
- `allocate_t1_batch` / `allocate_t2_batch` pop batches from each pool.
- The static `share_elites(a, b, top_pct=0.2)` performs a *pairwise* swap: A's worst are replaced by B's elites, and vice versa.

### `PopulationManager` (`population.py`)

Generates `total_size` random individuals and evaluates them sequentially.

---

## 8. How much of the bandit and helpers the controller actually uses

- **`UCB1Bandit`.** The controller instantiates it and, at Step 4, calls `register_algorithm` and `update`. It **never calls `select_next`**. All promotion and swap decisions use the controller's own UCB-style score from §3 (composite plus `0.1·sqrt(ln N / n_i)`). So the "UCB1 bandit" in EvoSynth is that inline formula. The `UCB1Bandit` class with `c = √2` is bookkeeping that the decision logic does not use.
- **`TierManager`.** Imported but not used. The controller implements the split itself, without shuffling: the sorted top 40% go to T1. It also implements elite sharing as *pooled* sharing, not `TierManager`'s pairwise swap.
- **`PopulationManager`.** Not used. The controller generates the population itself, in parallel.

---

## 9. Where the docs, paper or slides differ from the code

| Topic | Code (authoritative) | Elsewhere |
|---|---|---|
| Composite weights | 5 metrics: 0.30 / 0.25 / 0.15 / 0.15 / 0.15 | Paper §II-D: 4 pillars 0.40 / 0.20 / 0.20 / 0.20. The slides and the Hybrid Workflow use the code's 5-metric weights. |
| Exploration constant | `c = 0.1` | Panel guide says `c = 0.5`. The `UCB1Bandit` default is 1.414. The passive ablation telemetry uses 0.5. |
| Diversity | std/spread of fitness values | Panel guide: mean pairwise Euclidean distance in latent space |
| Speed | Equals trend (improvement) | Panel guide: inverse runtime. The ablation scripts' passive telemetry uses `1/(step_time+1e-4)`. |
| Efficiency | Constant 1.0 | Panel guide: `F/ln(t+e)`. The ablation scripts' passive telemetry uses that formula. |
| SWAP rule | `UCB_T1 > UCB_T2 + 0.10` (additive) | Several docs: "beats by >10%" or `UCB_T1 > 1.10 × UCB_T2` |
| Who gets the cooldown on SWAP | The demoted T2 family (`2k`). On rollback, the failed challenger (`2k`). | Docs: "demoted algorithm locked out" — consistent |
| T1 seeding | Contiguous rank slices of the sorted population; GA gets the best | Hybrid Workflow: interleaved ranks (I1, I5, I9…) |
| T2 reserve | The worst 60% of the initial population | Some docs describe it as neutral "reserved capacity" |
| Elite sharing | Pooled top-n, written into both islands | `TierManager.share_elites`: pairwise swap |
| Stagnation rule | T1: any zero-trend checkpoint → 50%. T2: 3 consecutive → 20%. | Docs: "3 consecutive checkpoints" for both |
| Default `t1_iters` / `k` | 30 / 10 in the constructor. Scripts override them (e.g. 3 / 2 for demos). | Slides: 10 / 5 |
