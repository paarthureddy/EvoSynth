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
    config = Configuration(space, vector=valid_vec)
    return {p: config[p] for p in list(space.keys())}


def _make_optimizer(algo_key, space, pop_size):
    if algo_key == 'GA':
        return GeneticAlgorithm(space, pop_size)
    return ParticleSwarm(space, pop_size)


class DynamicOptimizer:
    """
    Architecture:
      Step 0: Generate full population (N=50), evaluate all -> baseline scores
      Step 1: Split 40% -> T1 (GA + PSO compete for t1_iters)
              60% -> T2 (RESERVED, waiting for winner)
      Step 2: Winner promoted to T2 with the reserved 60%
              Losers keep evolving in T1
    """

    def __init__(self, search_space, evaluator,
                 budget=700, pop_size=50,
                 t1_iters=20, k_iters=5,
                 swap_threshold=0.10, stagnation_limit=3,
                 elite_pct=0.20, inject_pct=0.20):
        self.space = search_space
        self.eval = evaluator
        self.budget = budget
        self.pop_size = pop_size
        self.t1_iters = t1_iters
        self.k = k_iters
        self.swap_threshold = swap_threshold
        self.stagnation_limit = stagnation_limit
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
        n_inject = max(1, int(len(optimizer.population) * inject_pct))
        print(f"\n  [INJECT] Replacing bottom {n_inject} individuals in {tag}"
              f" with fresh random vectors.")
        fresh = self._evaluate_batch(
            [np.random.rand(dims) for _ in range(n_inject)], tag)
        optimizer.population.sort(key=lambda x: x.fitness, reverse=True)
        optimizer.population = optimizer.population[:-n_inject] + fresh
        return optimizer

    # ------------------------------------------------------------------
    def run(self):
        total_start = time.time()
        dims = len(list(self.space.keys()))

        # ==============================================================
        # STEP 0: Search Space Definition
        # ==============================================================
        print(_sep())
        print(_header("STEP 0: SEARCH SPACE DEFINITION"))
        print(_sep())
        print(f"  Hyperparameter Space : {dims} dimensions")
        for hp_name in list(self.space.keys()):
            print(f"    * {hp_name}: {self.space[hp_name]}")
        print(f"\n  Population Size      : {self.pop_size}")
        print(f"  Budget               : {self.budget} evals")
        print(f"  T1 Competition       : {self.t1_iters} iters")
        print(f"  T2 Scoring Interval  : {self.k} iters")
        print(f"  Swap Threshold       : {self.swap_threshold*100:.0f}% gap")
        print(f"  Stagnation Limit     : {self.stagnation_limit} checks")

        # ==============================================================
        # STEP 1: Generate & Evaluate FULL Population (N=50)
        # ==============================================================
        print(f"\n{_sep()}")
        print(_header(f"STEP 1: INITIAL POPULATION (N={self.pop_size})"))
        print(_sep())
        print(f"  Generating {self.pop_size} random configurations...")
        print(f"  Evaluating ALL with 3-Fold Cross-Validation...\n")

        full_pop = []
        for i in range(self.pop_size):
            if self.evals_used >= self.budget:
                break
            inds = self._evaluate_batch([np.random.rand(dims)], 'Init')
            full_pop.extend(inds)
            # Print progress every 10
            if (i + 1) % 10 == 0:
                best_so_far = max(ind.fitness for ind in full_pop)
                print(f"    Evaluated {i+1}/{self.pop_size}  "
                      f"(best so far: {best_so_far:.2f}%)")

        full_pop.sort(key=lambda x: x.fitness, reverse=True)
        fitnesses = [ind.fitness for ind in full_pop]
        print(f"\n  [OK] Full population evaluated: {len(full_pop)} individuals")
        print(f"       Best   : {fitnesses[0]:.2f}%")
        print(f"       Top-5  : " + "  ".join(f"{f:.2f}%" for f in fitnesses[:5]))
        print(f"       Avg    : {sum(fitnesses)/len(fitnesses):.2f}%")
        print(f"       Worst  : {fitnesses[-1]:.2f}%")
        print(f"       Budget : {self.evals_used}/{self.budget}")

        # ==============================================================
        # STEP 2: Tier Split (40% T1 / 60% T2 Reserved)
        # ==============================================================
        print(f"\n{_sep()}")
        print(_header("STEP 2: TIER SPLIT (40/60)"))
        print(_sep())

        # Sort by fitness, split top 40% to T1 (competition), bottom 60% reserved for T2
        t1_count = int(len(full_pop) * 0.4)
        t2_count = len(full_pop) - t1_count
        t1_pool = full_pop[:t1_count]    # Top 40% compete in T1
        t2_reserved = full_pop[t1_count:]  # Bottom 60% reserved for winner

        ga_size = t1_count // 2
        pso_size = t1_count - ga_size
        ga_seed = t1_pool[:ga_size]
        pso_seed = t1_pool[ga_size:]

        print(f"  Total Population : {len(full_pop)} individuals")
        print(f"  -----------------------------------------------")
        print(f"  Tier 1 (40%)     : {t1_count} individuals --> Competition Arena")
        print(f"    |-- GA  Island : {ga_size} individuals")
        print(f"    `-- PSO Island : {pso_size} individuals")
        print(f"  Tier 2 (60%)     : {t2_count} individuals --> RESERVED (waiting for winner)")

        # ==============================================================
        # STEP 3: Tier 1 Competition (GA vs PSO)
        # ==============================================================
        print(f"\n{_sep()}")
        print(_header("STEP 3: TIER 1 COMPETITION (GA vs PSO)"))
        print(_sep())
        print(f"  Both algorithms compete for {self.t1_iters} iterations.")
        print(f"  Winner (higher composite score) gets PROMOTED to Tier 2.\n")

        t1_ga = GeneticAlgorithm(self.space, ga_size)
        t1_pso = ParticleSwarm(self.space, pso_size)
        t1_ga.tell(ga_seed)
        t1_pso.tell(pso_seed)

        print(f"  {'Iter':>4}  {'GA Best':>9}  {'PSO Best':>10}  "
              f"{'Global':>9}  {'Time':>6}")
        print(f"  {'-'*4}  {'-'*9}  {'-'*10}  {'-'*9}  {'-'*6}")

        for t1_iter in range(1, self.t1_iters + 1):
            if self.evals_used >= self.budget:
                break
            ts = time.time()
            prev = self.best_individual.fitness if self.best_individual else 0.0

            ga_inds = self._evaluate_batch(t1_ga.ask(ga_size), 'T1_GA')
            pso_inds = self._evaluate_batch(t1_pso.ask(pso_size), 'T1_PSO')
            if ga_inds: t1_ga.tell(ga_inds)
            if pso_inds: t1_pso.tell(pso_inds)

            ga_b = max((i.fitness for i in t1_ga.population), default=0.0)
            pso_b = max((i.fitness for i in t1_pso.population), default=0.0)
            glob = self.best_individual.fitness if self.best_individual else 0.0
            flag = "  <-- NEW BEST" if glob > prev else ""
            print(f"  {t1_iter:>4}  {ga_b:>8.2f}%  {pso_b:>9.2f}%  "
                  f"{glob:>8.2f}%  {time.time()-ts:>5.1f}s{flag}")

        # ==============================================================
        # STEP 4: Composite Scoring + Promotion
        # ==============================================================
        print(f"\n{_sep()}")
        print(_header("STEP 4: COMPOSITE SCORING + PROMOTION"))
        print(_sep())

        t1_pops = {'T1_GA': list(t1_ga.population), 'T1_PSO': list(t1_pso.population)}
        scores, raw = self._compute_composite(t1_pops)
        self._print_scoring_table(t1_pops, scores, raw)

        winner_name = max(scores, key=scores.get)
        loser_name = [n for n in t1_pops if n != winner_name][0]
        winner_key = winner_name.replace("T1_", "")
        loser_key = loser_name.replace("T1_", "")

        print(f"\n  WINNER  : [{winner_name}] composite={scores[winner_name]:.3f}")
        print(f"  PROMOTE : [{winner_name}] --> Tier 2 (takes the {t2_count} reserved individuals)")
        print(f"  RETAIN  : [{loser_name}] stays in Tier 1 (keeps its {len(t1_pops[loser_name])} individuals)")

        self.bandit.register_algorithm('GA')
        self.bandit.register_algorithm('PSO')
        self.bandit.update(winner_key, scores[winner_name])
        self.bandit.update(loser_key, scores[loser_name])

        # Build T2: winner's population + the reserved 60%
        t2_combined = list(t1_pops[winner_name]) + t2_reserved
        t2_size = len(t2_combined)
        t2_optim = _make_optimizer(winner_key, self.space, t2_size)
        t2_optim.tell(t2_combined)
        t2_algo_key = winner_key

        # T1 loser keeps its population
        t1_optim = t1_pso if loser_key == 'PSO' else t1_ga
        t1_algo_key = loser_key
        t1_pop_size = len(t1_pops[loser_name])

        stagnation_counter = 0

        # ==============================================================
        # STEP 5: Tier 2 Working Loop + Switching Strategies
        # ==============================================================
        print(f"\n{_sep()}")
        print(_header(f"STEP 5: TIER 2 [{t2_algo_key}] + TIER 1 [{t1_algo_key}]"))
        print(_sep())
        print(f"  T2 [{t2_algo_key}] : {t2_size} individuals (winner + reserved 60%)")
        print(f"  T1 [{t1_algo_key}] : {t1_pop_size} individuals (challenger)")
        print(f"\n  Switching Strategies:")
        print(f"    a) Elite Sharing     -- every {self.k} iters")
        print(f"    b) Swap              -- if T1 beats T2 by >{self.swap_threshold*100:.0f}%")
        print(f"    c) Random Injection  -- if T2 stagnates {self.stagnation_limit}x")
        print(f"    d) UCB1 Force Swap   -- exploration bonus\n")

        print(f"  {'Iter':>4}  {'Status':<15}  {'T2 Best':>9}  {'T1 Best':>9}  "
              f"{'Global':>9}  {'Evals':>9}")
        print(f"  {'-'*4}  {'-'*15}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")

        iteration = 0
        while self.evals_used < self.budget:
            iteration += 1
            prev_best = self.best_individual.fitness if self.best_individual else 0.0

            # Evolve T2 (winner works on the large pool)
            t2_inds = self._evaluate_batch(
                t2_optim.ask(t2_size), f"T2_{t2_algo_key}")
            if t2_inds: t2_optim.tell(t2_inds)

            # Evolve T1 (loser keeps competing)
            t1_inds = self._evaluate_batch(
                t1_optim.ask(t1_pop_size), f"T1_{t1_algo_key}")
            if t1_inds: t1_optim.tell(t1_inds)

            if self.evals_used >= self.budget:
                break

            glob = self.best_individual.fitness if self.best_individual else 0.0
            t2_b = max((i.fitness for i in t2_optim.population), default=0.0)
            t1_b = max((i.fitness for i in t1_optim.population), default=0.0)
            new_b = glob > prev_best
            status = "*** NEW BEST ***" if new_b else "               "

            print(f"  {iteration:>4}  {status}  {t2_b:>8.2f}%  {t1_b:>8.2f}%  "
                  f"{glob:>8.2f}%  {self.evals_used:>4}/{self.budget}")
            if new_b:
                params = _decode_individual(self.best_individual, self.space)
                print("         `-- " + "  ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in params.items()))

            # ----------------------------------------------------------
            # Every k iterations: Switching Strategy Checkpoint
            # ----------------------------------------------------------
            if iteration % self.k == 0:
                print(f"\n  {_sep('-', 58)}")
                print(f"  {_header('SWITCHING CHECKPOINT', '-', 58)}")
                print(f"  {_sep('-', 58)}")

                all_pops = {f"T2_{t2_algo_key}": list(t2_optim.population),
                            f"T1_{t1_algo_key}": list(t1_optim.population)}
                sc, rw = self._compute_composite(all_pops)
                self._print_scoring_table(all_pops, sc, rw)

                t2_sc = sc.get(f"T2_{t2_algo_key}", 0.0)
                t1_sc = sc.get(f"T1_{t1_algo_key}", 0.0)
                self.bandit.update(t2_algo_key, t2_sc)
                self.bandit.update(t1_algo_key, t1_sc)
                ucb_choice = self.bandit.select_next([t2_algo_key, t1_algo_key])

                action_taken = False

                # Strategy b: Swap if T1 beats T2 by threshold
                if t1_sc > t2_sc + self.swap_threshold:
                    print(f"\n  [SWAP] T1 ({t1_sc:.3f}) beats T2 ({t2_sc:.3f})"
                          f" by >{self.swap_threshold*100:.0f}%!")
                    print(f"  [SWAP] Demoting [{t2_algo_key}] to T1."
                          f" Promoting [{t1_algo_key}] to T2.")
                    old_t2 = list(t2_optim.population)
                    old_t1 = list(t1_optim.population)
                    new_t2 = _make_optimizer(t1_algo_key, self.space, t2_size)
                    new_t2.tell(old_t1 + old_t2[len(old_t1):])  # T1 pop + remaining T2
                    new_t1 = _make_optimizer(t2_algo_key, self.space, t1_pop_size)
                    new_t1.tell(old_t2[:t1_pop_size])
                    t2_optim, t1_optim = new_t2, new_t1
                    t2_algo_key, t1_algo_key = t1_algo_key, t2_algo_key
                    stagnation_counter = 0
                    action_taken = True

                # Strategy c: Stagnation -> injection
                elif rw.get(f"T2_{t2_algo_key}", {}).get('trend', 1.0) == 0.0:
                    stagnation_counter += 1
                    print(f"\n  [STAGNATION] T2 trend=0 for {stagnation_counter}/"
                          f"{self.stagnation_limit} checks.")
                    if stagnation_counter >= self.stagnation_limit:
                        t2_optim = self._random_inject(
                            t2_optim, self.inject_pct, dims, f"T2_{t2_algo_key}")
                        stagnation_counter = 0
                        action_taken = True
                else:
                    stagnation_counter = 0

                # Strategy d: UCB1 force swap
                if not action_taken and ucb_choice == t1_algo_key:
                    print(f"\n  [UCB1 FORCE] Exploration bonus favors [{t1_algo_key}]!")
                    old_t2 = list(t2_optim.population)
                    old_t1 = list(t1_optim.population)
                    new_t2 = _make_optimizer(t1_algo_key, self.space, t2_size)
                    new_t2.tell(old_t1 + old_t2[len(old_t1):])
                    new_t1 = _make_optimizer(t2_algo_key, self.space, t1_pop_size)
                    new_t1.tell(old_t2[:t1_pop_size])
                    t2_optim, t1_optim = new_t2, new_t1
                    t2_algo_key, t1_algo_key = t1_algo_key, t2_algo_key
                    stagnation_counter = 0
                    action_taken = True

                # Strategy a: Elite sharing (always)
                print(f"\n  [ELITE SHARE] Top {int(self.elite_pct*100)}%"
                      f" swapped: T2_{t2_algo_key} <-> T1_{t1_algo_key}")
                n_share = max(1, int(min(len(t2_optim.population),
                                        len(t1_optim.population)) * self.elite_pct))
                t2_s = sorted(t2_optim.population, key=lambda x: x.fitness, reverse=True)
                t1_s = sorted(t1_optim.population, key=lambda x: x.fitness, reverse=True)
                t2_optim.population = t2_s[:-n_share] + t1_s[:n_share]
                t1_optim.population = t1_s[:-n_share] + t2_s[:n_share]

                print(f"  {_sep('-', 58)}\n")
                print(f"  {'Iter':>4}  {'Status':<15}  {'T2 Best':>9}  {'T1 Best':>9}  "
                      f"{'Global':>9}  {'Evals':>9}")
                print(f"  {'-'*4}  {'-'*15}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")

        # ==============================================================
        # FINAL RESULTS
        # ==============================================================
        total_elapsed = time.time() - total_start
        print(f"\n{_sep()}")
        print(_header("FINAL RESULTS"))
        print(_sep())
        print(f"  Total Evaluations  : {self.evals_used}")
        print(f"  Total Time         : {total_elapsed:.1f}s")
        print(f"  Winning Algorithm  : {self.best_individual.source_algorithm}")
        print(f"  Best Accuracy      : {self.best_individual.fitness:.4f}%")
        print(f"\n  Best Hyperparameters:")
        params = _decode_individual(self.best_individual, self.space)
        for k, v in params.items():
            if isinstance(v, float):
                print(f"    * {k:<20}: {v:.4f}")
            else:
                print(f"    * {k:<20}: {v}")
        print(_sep())

        return self.best_individual
