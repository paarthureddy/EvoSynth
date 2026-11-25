# 04 — The Four Optimizers

All optimizers live in `src/evosynth/optimizers/`; their constants are in `settings/optimizers.toml`. They operate **only** on latent vectors in `[0,1]^d` and never see real hyperparameter values.

---

## 1. The ask/tell contract (`base.py`)

```python
class BaseOptimizer:
    def __init__(self, search_space, pop_size):
        self.search_space = search_space
        self.pop_size = pop_size
        self.dimensions = len(search_space.get_hyperparameters())
        self.population = []          # list[Individual], kept sorted best-first by subclasses

    def ask(self, num_samples):       # -> list[np.ndarray] in [0,1]^d
        raise NotImplementedError

    def tell(self, evaluated_individuals):   # Individuals with .fitness set
        raise NotImplementedError
```

The controller relies on four things:

1. `ask(n)` returns `n` vectors.
2. `tell(list)` absorbs evaluated individuals. **The first `tell` is used to seed the optimizer** with an existing population: the T1 seeds, or T1 + reserve for T2, or the swap populations.
3. `.population` holds the current island, sorted best-first and truncated to `pop_size`.
4. `.pop_size` is the island size. In Tier 2, `ask(opt.pop_size)` is called each iteration.

Fitness is **maximised** (accuracy in %).

---

## 2. Genetic Algorithm (`ga.py`)

| Item | Implementation |
|---|---|
| Constructor | `GeneticAlgorithm(space, pop_size, mutation_rate=0.1)` |
| Cold start | If the population has fewer than 2 members, `ask` returns uniform random vectors |
| Parent selection | Binary tournament, done twice: sample 2 random members and keep the fitter one as p1; repeat for p2 |
| Crossover | Continuous arithmetic crossover with a per-gene random weight: `alpha ~ U(0,1)^d`, `child = alpha·p1 + (1−alpha)·p2` |
| Mutation | Each gene independently, with probability 0.1, gets `N(0, 0.1)` noise added |
| Bounds | `np.clip(child, 0, 1)` |
| Survivor selection (`tell`) | (μ+λ) elitism: append the evaluated offspring to the population, sort by fitness, keep the top `pop_size` |

**Character:** global exploration by blending parents, with light Gaussian mutation. Strong elitism means the best individual is never lost.

---

## 3. Particle Swarm Optimization (`pso.py`)

| Item | Implementation |
|---|---|
| Constants | Inertia `w = 0.5`, cognitive `c1 = 1.5`, social `c2 = 1.5` |
| Particle state | `pos`, `vel`, `pbest_pos`, `pbest_score`. The swarm holds `gbest_pos` and `gbest_score`. |
| Seeding | The first `tell` (while there are no particles) only stores the individuals in `self.population`. On the first `ask`, `_seed_particles_from_population()` turns each stored individual into a particle (its position is the vector, its pbest is its fitness, its velocity is `U(−0.1, 0.1)`). It then pads with random particles up to `pop_size` and sets gbest from the seeds. |
| First `ask` after seeding | Returns the **current positions** of the first `n` particles (no movement). These are re-evaluated, which uses budget once. |
| Velocity update | `v ← w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)`, with `r1, r2 ~ U(0,1)^d` |
| Position update | `x ← clip(x + v, 0, 1)`. Velocity is not clamped. |
| Particle indexing | `ask(n)` moves particle `i % n_particles` for `i = 0..n−1` |
| `tell` | Maps the i-th evaluated individual to particle i: sets `pos` to the evaluated vector, and updates pbest and gbest on improvement. Then merges into `self.population`, sorts, and truncates to `pop_size`. |
| Pure cold start | With no seed population, it creates `pop_size` random particles |

**Character:** fast, momentum-driven convergence towards the personal and global bests. It is the most exploitative optimizer and the most prone to collapse on noisy landscapes.

---

## 4. Differential Evolution (`de.py`)

| Item | Implementation |
|---|---|
| Constants | Mutation factor `F = 0.8`, crossover rate `CR = 0.7` |
| Cold start | With fewer than 3 population members, returns random vectors |
| Strategy | **DE/rand/1/bin** |
| Target vector | `x_i = population[i % pop_len]`, cycling through the population |
| Donor | Choose 3 distinct indices `a, b, c ≠ i`. If fewer than 3 candidates exist, sample with replacement. Then `v = clip(a + F·(b − c), 0, 1)`. |
| Crossover | Binomial: `u = copy(x_i)`. For each gene j, `u_j = v_j` if `rand < CR` or `j == j_rand` (one gene is guaranteed to come from the donor). |
| Selection (`tell`) | (μ+λ) truncation: merge, sort, keep the top `pop_size`. This is a population-level greedy selection, not the classic one-to-one parent-vs-trial replacement. |

**Character:** robust exploration by vector differences that scale naturally with the population's spread. In the CVGS-300 ablation, DE kept the highest final diversity of the isolated optimizers (0.475).

---

## 5. CMA-ES (`cmaes.py`)

Wraps the `cma` library's `CMAEvolutionStrategy`.

| Item | Implementation |
|---|---|
| Lazy initialisation | Happens on the first `ask`. The mean is the population mean if a seed population exists, else `0.5·1`. |
| Settings | `sigma0 = 0.25`, `bounds = [0, 1]`, `popsize = max(4, pop_size)`, `verbose = −9` |
| `ask(n)` | Draws from a buffer (`pending_solutions`). When the buffer is empty it calls `es.ask()` for a full generation, then hands out vectors one at a time. Results are clipped to `[0, 1]`. |
| `tell` | Buffers `(vector, −fitness)` pairs, because CMA-ES minimises. Whenever the buffer holds at least one full generation (`es.popsize`), it calls `es.tell(batch_x, batch_y)`. Exceptions (numerical singularities near tight bounds) are swallowed. |
| Seed handling | The seeding `tell` occurs before `es` exists, so the seed individuals go into the buffer and into `self.population`. The buffered seeds are fed to `es.tell` on the first `tell` call after `es` has been created. |
| Population tracking | Merge, sort, keep the top `pop_size`. This is used for scoring, sharing and injection. |

**Character:** adapts a full covariance matrix over the d latent dimensions, which suits fine-tuning correlated continuous parameters. On Covertype at 300 and 500 evaluations, **CMA-ES (in Tier 2) found the winning configuration**.

---

## 6. Side-by-side

| | GA | PSO | DE | CMA-ES |
|---|---|---|---|---|
| Family | Evolutionary | Swarm | Evolutionary (vector differences) | Evolution strategy |
| Key constants | mutation p = 0.1, σ = 0.1 | w = 0.5, c1 = c2 = 1.5 | F = 0.8, CR = 0.7 | σ0 = 0.25 |
| Uses fitness ranks | Tournament (size 2) | pbest / gbest | Truncation selection | Weighted recombination (inside `cma`) |
| Memory across iterations | Population | Particles (pos, vel, pbest) + gbest | Population | Mean, covariance, step size |
| Warm start from seeds | Yes (population) | Yes (particles built from seeds) | Yes (population) | Yes (mean = seed mean; seeds told later) |
| Survivor selection | (μ+λ) | Particle replacement + tracked top population | (μ+λ) | CMA update + tracked top population |
| Typical role | Broad exploration | Fast exploitation | Robust exploration | Fine-tuning correlated dimensions |

The panel material's "NFL" summary of each optimizer: GA explores globally but converges slowly; PSO accelerates quickly but converges prematurely; DE navigates robustly but is sensitive to diversity collapse; CMA-ES handles ill-conditioned problems well but is heavier.

---

## 7. Adding a new optimizer

1. Create `src/evosynth/optimizers/my_opt.py` (constants in `settings/optimizers.toml`) with a subclass of `BaseOptimizer` that implements `ask(n)` and `tell(inds)`:
   - keep `self.population` sorted best-first and truncated to `self.pop_size`;
   - treat the first `tell` as a seed population;
   - always return vectors clipped to `[0, 1]`.
2. Export it in `optimizers/__init__.py`.
3. In `controller.py`:
   - import it and add a branch to `_make_optimizer`, e.g. `if algo_key == 'MYOPT': return MyOpt(space, pop_size)`;
   - add `'MYOPT'` to `self.algo_evals`, `self.cooldowns` and `self._algo_stats`.
4. **Watch the structural assumptions.** The current tier logic hard-codes four families:
   - the Step 2 `seeds` dict splits Tier 1 into **4** slices;
   - Step 4 promotes exactly 2 and keeps exactly 2.

   A fifth optimizer therefore needs the split (`t1_count // 4` and the seeds dict) and the `top2` / `bot2` logic generalised.
