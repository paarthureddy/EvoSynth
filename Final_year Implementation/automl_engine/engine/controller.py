import numpy as np
import time
from automl_engine.engine.tier_manager import TierManager
from automl_engine.engine.composite import CompositeScoringEngine
from automl_engine.engine.ucb1 import UCB1Bandit
from automl_engine.optimizers.ga import GeneticAlgorithm
from automl_engine.optimizers.pso import ParticleSwarm
from automl_engine.core.individual import Individual


def _sep(char="=", width=62):
    return char * width


def _header(title, char="=", width=62):
    side = (width - len(title) - 2) // 2
    return f"{char * side} {title} {char * side}"


def _decode_individual(individual, space):
    import ConfigSpace.hyperparameters as CSH
    from ConfigSpace import Configuration
    valid_vec = np.copy(individual.vector)
    for i, hp_name in enumerate(list(space.keys())):
        hp = space[hp_name]
        if isinstance(hp, CSH.CategoricalHyperparameter):
            num_choices = len(hp.choices)
            idx = int(np.floor(valid_vec[i] * num_choices))
            idx = min(idx, num_choices - 1)
            valid_vec[i] = float(idx)
    from ConfigSpace import Configuration
    config = Configuration(space, vector=valid_vec)
    return {p: config[p] for p in list(space.keys())}


def _make_optimizer(algo_key, space, pop_size):
    """Factory: returns a fresh GA or PSO instance."""
    if algo_key == 'GA':
        return GeneticAlgorithm(space, pop_size)
    return ParticleSwarm(space, pop_size)


class DynamicOptimizer:
    """
    Full hybrid controller implementing:
      1. Tier 1 competition (GA vs PSO for t1_iters)
      2. Composite scoring -> initial promotion
      3. Tier 2 main loop with:
         a. Elite sharing  (every k iters)
         b. Demotion/Promotion swap  (if T1 beats T2 by swap_threshold)
         c. Stagnation detection + random injection (if T2 flat for stagnation_limit checks)
         d. UCB1 exploration-driven force swap  (if exploration bonus fires)
    """

    def __init__(self, search_space, evaluator,
                 budget=300, t1_iters=20, k_iters=5,
                 swap_threshold=0.10, stagnation_limit=3,
                 elite_pct=0.20, inject_pct=0.20):
        self.space = search_space
        self.eval = evaluator
        self.budget = budget
        self.t1_iters = t1_iters
        self.k = k_iters
        self.swap_threshold = swap_threshold      # composite gap to trigger swap
        self.stagnation_limit = stagnation_limit  # consecutive flat checkpoints before injection
        self.elite_pct = elite_pct
        self.inject_pct = inject_pct

        self.composite_engine = CompositeScoringEngine()
        self.bandit = UCB1Bandit()

        self.evals_used = 0
        self.best_individual = None
        self._algo_stats = {
            'GA':  {'best': 0.0, 'prev_best': 0.0},
            'PSO': {'best': 0.0, 'prev_best': 0.0},
        }

    # ------------------------------------------------------------------
    def _evaluate_batch(self, vectors, algo_name):
        algo_key = algo_name.replace("T2_", "").replace("T1_", "")
        inds = []
        for vec in vectors:
            if self.evals_used >= self.budget:
                break
            ind = Individual(vec)
            fitness = self.eval.evaluate(ind)
            ind.set_fitness(fitness)
            ind.source_algorithm = algo_name
            inds.append(ind)
            self.evals_used += 1
            if algo_key in self._algo_stats:
                if fitness > self._algo_stats[algo_key]['best']:
                    self._algo_stats[algo_key]['best'] = fitness
            if self.best_individual is None or fitness > self.best_individual.fitness:
                self.best_individual = ind
        return inds

    # ------------------------------------------------------------------
    def _compute_composite(self, algo_pops):
        algo_data = {}
        for name, pop in algo_pops.items():
            if not pop:
                continue
            fitnesses = [ind.fitness for ind in pop]
            best = max(fitnesses)
            spread = max(fitnesses) - min(fitnesses) + 1e-9
            diversity = min(float(np.std(fitnesses)) / spread, 1.0)
            algo_key = name.replace("T1_", "").replace("T2_", "")
            prev = self._algo_stats.get(algo_key, {}).get('prev_best', 0.0)
            trend = max(best - prev, 0.0)
            self._algo_stats[algo_key]['prev_best'] = best
            algo_data[name] = {
                'fitness': best, 'speed': trend,
                'efficiency': 1.0, 'diversity': diversity, 'trend': trend
            }
        return self.composite_engine.calculate_scores(algo_data), algo_data

    # ------------------------------------------------------------------
    def _print_scoring_table(self, algo_pops, scores, raw):
        print(f"\n  {'Algorithm':<12} {'Best Acc':>9} {'Diversity':>10} {'Composite':>10}")
        print(f"  {'-'*12} {'-'*9} {'-'*10} {'-'*10}")
        for name in algo_pops:
            if name not in raw:
                continue
            print(f"  {name:<12} {raw[name]['fitness']:>8.2f}%"
                  f" {raw[name]['diversity']:>10.3f}"
                  f" {scores.get(name, 0.0):>10.3f}")

    # ------------------------------------------------------------------
    def _random_inject(self, optimizer, inject_pct, dims, tag):
        """Replaces the bottom inject_pct of an optimizer's population with fresh random individuals."""
        n_inject = max(1, int(len(optimizer.population) * inject_pct))
        print(f"\n  [INJECT] Replacing bottom {n_inject} individuals in {tag} with fresh random vectors.")
        fresh_vecs = [np.random.rand(dims) for _ in range(n_inject)]
        fresh_inds = self._evaluate_batch(fresh_vecs, tag)
        # Replace worst
        optimizer.population.sort(key=lambda x: x.fitness, reverse=True)
        optimizer.population = optimizer.population[:-n_inject] + fresh_inds
        return optimizer

    # ------------------------------------------------------------------
    def run(self):
        total_start = time.time()
        dims = len(list(self.space.keys()))

        # ==============================================================
        # STEP 1: Search Space
        # ==============================================================
        print(_sep())
        print(_header("STEP 1: SEARCH SPACE DEFINITION"))
        print(_sep())
        print(f"  Hyperparameter Space : {dims} dimensions")
        for hp_name in list(self.space.keys()):
            print(f"    * {hp_name}: {self.space[hp_name]}")
        print(f"\n  Budget             : {self.budget} evals")
        print(f"  T1 Competition     : {self.t1_iters} iters")
        print(f"  Scoring Interval k : {self.k} iters")
        print(f"  Swap Threshold     : {self.swap_threshold*100:.0f}% composite gap")
        print(f"  Stagnation Limit   : {self.stagnation_limit} consecutive flat checks")

        # ==============================================================
        # STEP 2: Warm-Up
        # ==============================================================
        print(f"\n{_sep()}")
        print(_header("STEP 2: WARM-UP (10 Random Individuals)"))
        print(_sep())
        print("  Generating 10 random configs and evaluating...\n")
        warmup_pop = []
        for _ in range(10):
            if self.evals_used >= self.budget:
                break
            warmup_pop.extend(self._evaluate_batch([np.random.rand(dims)], 'RandomInit'))
        fitnesses = sorted([ind.fitness for ind in warmup_pop], reverse=True)
        print(f"  [OK] {len(warmup_pop)} individuals evaluated")
        print(f"       Top-3  : {fitnesses[0]:.2f}%  {fitnesses[1]:.2f}%  {fitnesses[2]:.2f}%")
        print(f"       Avg    : {sum(fitnesses)/len(fitnesses):.2f}%")
        print(f"       Budget : {self.evals_used}/{self.budget}")

        # ==============================================================
        # STEP 3: Tier Split
        # ==============================================================
        print(f"\n{_sep()}")
        print(_header("STEP 3: TIER SPLIT"))
        print(_sep())
        t1_size = max(len(warmup_pop) // 2, 2)
        ga_seed  = warmup_pop[:t1_size]
        pso_seed = warmup_pop[t1_size:]
        print(f"  T1 GA  Island : {len(ga_seed)} individuals  --> Competition Arena")
        print(f"  T1 PSO Island : {len(pso_seed)} individuals --> Competition Arena")
        print(f"  T2            : EMPTY  (promoted winner fills this)")

        # ==============================================================
        # STEP 4: Tier 1 Competition
        # ==============================================================
        print(f"\n{_sep()}")
        print(_header("STEP 4: TIER 1 COMPETITION (GA vs PSO)"))
        print(_sep())
        print(f"  Competing for {self.t1_iters} iterations. Winner --> Tier 2.\n")

        t1_ga  = GeneticAlgorithm(self.space, len(ga_seed))
        t1_pso = ParticleSwarm(self.space, len(pso_seed))
        t1_ga.tell(ga_seed)
        t1_pso.tell(pso_seed)

        print(f"  {'Iter':>4}  {'GA Best':>9}  {'PSO Best':>10}  {'Global':>9}  {'Time':>6}")
        print(f"  {'-'*4}  {'-'*9}  {'-'*10}  {'-'*9}  {'-'*6}")

        for t1_iter in range(1, self.t1_iters + 1):
            if self.evals_used >= self.budget:
                break
            ts = time.time()
            prev = self.best_individual.fitness if self.best_individual else 0.0

            ga_inds  = self._evaluate_batch(t1_ga.ask(len(ga_seed)),   'T1_GA')
            pso_inds = self._evaluate_batch(t1_pso.ask(len(pso_seed)), 'T1_PSO')
            if ga_inds:  t1_ga.tell(ga_inds)
            if pso_inds: t1_pso.tell(pso_inds)

            ga_b  = max((i.fitness for i in t1_ga.population),  default=0.0)
            pso_b = max((i.fitness for i in t1_pso.population), default=0.0)
            glob  = self.best_individual.fitness if self.best_individual else 0.0
            flag  = "  <-- NEW BEST" if glob > prev else ""
            print(f"  {t1_iter:>4}  {ga_b:>8.2f}%  {pso_b:>9.2f}%  "
                  f"{glob:>8.2f}%  {time.time()-ts:>5.1f}s{flag}")

        # ==============================================================
        # STEP 5: Composite Scoring -> Initial Promotion
        # ==============================================================
        print(f"\n{_sep()}")
        print(_header("STEP 5: COMPOSITE SCORING + INITIAL PROMOTION"))
        print(_sep())

        t1_pops = {'T1_GA': list(t1_ga.population), 'T1_PSO': list(t1_pso.population)}
        scores, raw = self._compute_composite(t1_pops)
        self._print_scoring_table(t1_pops, scores, raw)

        winner_name = max(scores, key=scores.get)
        loser_name  = [n for n in t1_pops if n != winner_name][0]
        winner_key  = winner_name.replace("T1_", "")
        loser_key   = loser_name.replace("T1_", "")

        print(f"\n  WINNER  : [{winner_name}] composite={scores[winner_name]:.3f}")
        print(f"  PROMOTE : [{winner_name}] --> Tier 2")
        print(f"  RETAIN  : [{loser_name}] stays in Tier 1 as challenger")

        # Register both in UCB1
        self.bandit.register_algorithm('GA')
        self.bandit.register_algorithm('PSO')
        self.bandit.update(winner_key, scores[winner_name])
        self.bandit.update(loser_key,  scores[loser_name])

        # Build T2 from winner's population
        t2_size = max(len(t1_pops[winner_name]), 5)
        t2_optim = _make_optimizer(winner_key, self.space, t2_size)
        t2_optim.tell(list(t1_pops[winner_name]))
        t2_algo_key = winner_key   # track which algo is in T2

        # Loser stays in T1
        t1_optim     = t1_pso if loser_key == 'PSO' else t1_ga
        t1_algo_key  = loser_key
        t1_pop_size  = len(t1_pops[loser_name])

        stagnation_counter = 0  # consecutive flat composite scoring events

        # ==============================================================
        # STEP 6: Tier 2 Main Loop
        # ==============================================================
        print(f"\n{_sep()}")
        print(_header(f"STEP 6: TIER 2 LOOP  [{t2_algo_key} in T2 | {t1_algo_key} in T1]"))
        print(_sep())
        print(f"  Switching Strategies Active:")
        print(f"    a) Elite Sharing     -- every {self.k} iters")
        print(f"    b) Swap              -- if T1 composite beats T2 by >{self.swap_threshold*100:.0f}%")
        print(f"    c) Random Injection  -- if T2 stagnates for {self.stagnation_limit} checks")
        print(f"    d) UCB1 Force Swap   -- if exploration bonus triggers\n")

        print(f"  {'Iter':>4}  {'Status':<18}  {'T2 Best':>9}  {'T1 Best':>9}  "
              f"{'Global':>9}  {'Evals':>9}")
        print(f"  {'-'*4}  {'-'*18}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")

        iteration = 0
        while self.evals_used < self.budget:
            iteration += 1
            prev_best = self.best_individual.fitness if self.best_individual else 0.0

            # Evolve T2
            t2_inds = self._evaluate_batch(t2_optim.ask(t2_size), f"T2_{t2_algo_key}")
            if t2_inds: t2_optim.tell(t2_inds)

            # Evolve T1
            t1_inds = self._evaluate_batch(t1_optim.ask(t1_pop_size), f"T1_{t1_algo_key}")
            if t1_inds: t1_optim.tell(t1_inds)

            if self.evals_used >= self.budget:
                break

            glob   = self.best_individual.fitness if self.best_individual else 0.0
            t2_b   = max((i.fitness for i in t2_optim.population), default=0.0)
            t1_b   = max((i.fitness for i in t1_optim.population), default=0.0)
            new_b  = glob > prev_best
            status = "*** NEW BEST ***  " if new_b else "                  "

            print(f"  {iteration:>4}  {status}  {t2_b:>8.2f}%  {t1_b:>8.2f}%  "
                  f"{glob:>8.2f}%  {self.evals_used:>4}/{self.budget}")
            if new_b:
                params = _decode_individual(self.best_individual, self.space)
                print("         `-- " + "  ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in params.items()))

            # ----------------------------------------------------------
            # Every k iterations: run all switching strategies
            # ----------------------------------------------------------
            if iteration % self.k == 0:
                print(f"\n  {_sep('-', 58)}")
                print(f"  {_header('SWITCHING STRATEGY CHECKPOINT', '-', 58)}")
                print(f"  {_sep('-', 58)}")

                all_pops = {f"T2_{t2_algo_key}": list(t2_optim.population),
                            f"T1_{t1_algo_key}": list(t1_optim.population)}
                sc, rw = self._compute_composite(all_pops)
                self._print_scoring_table(all_pops, sc, rw)

                t2_sc = sc.get(f"T2_{t2_algo_key}", 0.0)
                t1_sc = sc.get(f"T1_{t1_algo_key}", 0.0)

                # Update UCB1
                self.bandit.update(t2_algo_key, t2_sc)
                self.bandit.update(t1_algo_key, t1_sc)
                ucb_choice = self.bandit.select_next([t2_algo_key, t1_algo_key])

                action_taken = False

                # ── STRATEGY b: Demotion/Promotion Swap ───────────────
                if t1_sc > t2_sc + self.swap_threshold:
                    print(f"\n  [SWAP] T1 composite ({t1_sc:.3f}) beats T2 ({t2_sc:.3f}) "
                          f"by >{self.swap_threshold*100:.0f}% threshold!")
                    print(f"  [SWAP] Demoting [{t2_algo_key}] to T1. "
                          f"Promoting [{t1_algo_key}] to T2.")

                    # Swap populations and optimizers
                    old_t2_pop = list(t2_optim.population)
                    old_t1_pop = list(t1_optim.population)

                    new_t2_optim = _make_optimizer(t1_algo_key, self.space, t2_size)
                    new_t2_optim.tell(old_t1_pop)
                    new_t1_optim = _make_optimizer(t2_algo_key, self.space, t1_pop_size)
                    new_t1_optim.tell(old_t2_pop)

                    t2_optim, t1_optim = new_t2_optim, new_t1_optim
                    t2_algo_key, t1_algo_key = t1_algo_key, t2_algo_key
                    stagnation_counter = 0
                    action_taken = True

                # ── STRATEGY c: Stagnation Detection + Injection ───────
                elif rw.get(f"T2_{t2_algo_key}", {}).get('trend', 1.0) == 0.0:
                    stagnation_counter += 1
                    print(f"\n  [STAGNATION] T2 trend = 0 for {stagnation_counter}/"
                          f"{self.stagnation_limit} consecutive checks.")
                    if stagnation_counter >= self.stagnation_limit:
                        t2_optim = self._random_inject(
                            t2_optim, self.inject_pct, dims, f"T2_{t2_algo_key}")
                        stagnation_counter = 0
                        action_taken = True
                else:
                    stagnation_counter = 0

                # ── STRATEGY d: UCB1 Force Swap ────────────────────────
                if not action_taken and ucb_choice == t1_algo_key:
                    print(f"\n  [UCB1 FORCE] Exploration bonus favors [{t1_algo_key}]! "
                          f"Force-promoting it to T2.")
                    old_t2_pop = list(t2_optim.population)
                    old_t1_pop = list(t1_optim.population)
                    new_t2_optim = _make_optimizer(t1_algo_key, self.space, t2_size)
                    new_t2_optim.tell(old_t1_pop)
                    new_t1_optim = _make_optimizer(t2_algo_key, self.space, t1_pop_size)
                    new_t1_optim.tell(old_t2_pop)
                    t2_optim, t1_optim = new_t2_optim, new_t1_optim
                    t2_algo_key, t1_algo_key = t1_algo_key, t2_algo_key
                    stagnation_counter = 0
                    action_taken = True

                # ── STRATEGY a: Elite Sharing (always runs) ────────────
                print(f"\n  [ELITE SHARE] Swapping top {int(self.elite_pct*100)}% "
                      f"between T2_{t2_algo_key} <-> T1_{t1_algo_key}")
                n_share = max(1, int(min(len(t2_optim.population),
                                        len(t1_optim.population)) * self.elite_pct))
                t2_sorted = sorted(t2_optim.population, key=lambda x: x.fitness, reverse=True)
                t1_sorted = sorted(t1_optim.population, key=lambda x: x.fitness, reverse=True)
                # T2 gets T1's best; T1 gets T2's best
                t2_optim.population = t2_sorted[:-n_share] + t1_sorted[:n_share]
                t1_optim.population = t1_sorted[:-n_share] + t2_sorted[:n_share]

                print(f"  {_sep('-', 58)}\n")
                # Reprint header
                print(f"  {'Iter':>4}  {'Status':<18}  {'T2 Best':>9}  {'T1 Best':>9}  "
                      f"{'Global':>9}  {'Evals':>9}")
                print(f"  {'-'*4}  {'-'*18}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")

        # ==============================================================
        # STEP 7: Final Results
        # ==============================================================
        total_elapsed = time.time() - total_start
        print(f"\n{_sep()}")
        print(_header("STEP 7: FINAL RESULTS"))
        print(_sep())
        print(f"  Total Evaluations  : {self.evals_used}")
        print(f"  Total Time         : {total_elapsed:.1f}s")
        print(f"  Winning Algorithm  : {self.best_individual.source_algorithm}")
        print(f"  Best Accuracy      : {self.best_individual.fitness:.4f}%")
        print(f"\n  Best Hyperparameters Discovered:")
        params = _decode_individual(self.best_individual, self.space)
        for k, v in params.items():
            if isinstance(v, float):
                print(f"    * {k:<20}: {v:.4f}")
            else:
                print(f"    * {k:<20}: {v}")
        print(_sep())

        return self.best_individual
