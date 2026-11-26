"""
DynamicOptimizer: the EvoSynth meta-optimization loop (Steps 0-6).

Every default comes from the settings folder: [engine], [bandit], [safeguards] in
settings/engine.toml and [run] in settings/run.toml.
Events go to the active EventReporter (evosynth.telemetry.reporter.get_reporter), which a
run context points at the run's events.jsonl. Importing this module has no side effects:
the run folder, run.log and stdout mirror are set up by evosynth.experiments.run_context.
"""
import os
import time

import numpy as np

from evosynth.config import get_settings
from evosynth.core.decoding import decode_vector_to_dict
from evosynth.core.individual import Individual
from evosynth.engine.bandit import UCB1Bandit
from evosynth.engine.scoring import CompositeScoringEngine
from evosynth.engine.tiers import TierManager
from evosynth.optimizers.cmaes import CMAESOptimizer
from evosynth.optimizers.de import DifferentialEvolution
from evosynth.optimizers.ga import GeneticAlgorithm
from evosynth.optimizers.pso import ParticleSwarm
from evosynth.seeding import stream
from evosynth.telemetry.reporter import get_reporter
from evosynth.telemetry.run_logger import PopulationLogger


def _sep(char="=", width=80):
    return char * width


def _header(title, char="=", width=80):
    side = (width - len(title) - 2) // 2
    return f"{char * side} {title} {char * side}"


def _decode_individual(individual, space):
    return decode_vector_to_dict(individual.vector, space)


def _make_optimizer(algo_key, space, pop_size, rng=None):
    if algo_key == 'GA': return GeneticAlgorithm(space, pop_size, rng=rng)
    if algo_key == 'PSO': return ParticleSwarm(space, pop_size, rng=rng)
    if algo_key == 'DE': return DifferentialEvolution(space, pop_size, rng=rng)
    if algo_key == 'CMAES': return CMAESOptimizer(space, pop_size, rng=rng)
    raise ValueError(f"Unknown algo: {algo_key}")


def _pick(value, default):
    return default if value is None else value


class DynamicOptimizer:
    """Arguments left as None take their value from the active settings (settings/engine.toml, settings/run.toml)."""

    def __init__(self, search_space, evaluator,
                 budget=None, pop_size=None,
                 t1_iters=None, k_iters=None,
                 swap_margin=None, stagnation_limit=None,
                 elite_pct=None, inject_pct=None,
                 max_workers=None, population_csv=None):
        cfg = get_settings()
        eng, sg = cfg.engine, cfg.safeguards
        self.space = search_space
        self.eval = evaluator
        self.budget = _pick(budget, eng.budget)
        self.pop_size = _pick(pop_size, eng.pop_size)
        self.t1_iters = _pick(t1_iters, eng.t1_iters)
        self.k = _pick(k_iters, eng.k_iters)
        self.swap_margin = _pick(swap_margin, sg.swap_margin)   # additive UCB margin for a SWAP
        self.stagnation_limit = _pick(stagnation_limit, sg.stagnation_limit)
        self.elite_pct = _pick(elite_pct, sg.elite_pct)
        self.inject_pct = _pick(inject_pct, sg.t2_inject_pct)
        workers = _pick(max_workers, cfg.run.max_workers)
        self.max_workers = workers if workers else min(8, os.cpu_count() or 4)

        # Constants previously hard-coded in run()
        self.t1_fraction = eng.t1_fraction
        self.n_promoted = eng.n_promoted
        self.min_island_size = eng.min_island_size
        self.tier_split = eng.tier_split
        self.auto_k = eng.auto_k
        self.min_t2_checkpoints = eng.min_t2_checkpoints
        self.trend_window = cfg.composite.trend_window
        self.t1_inject_pct = sg.t1_inject_pct
        self.grace_iters = sg.grace_iters
        self.cooldown_multiplier = sg.cooldown_multiplier

        self.composite_engine = CompositeScoringEngine()
        # Phase 1.1: the bandit makes promotion and SWAP decisions (c from [bandit] c)
        self.bandit = UCB1Bandit()

        # Phase 0 seeding: the controller's own generator (initial population, injections) and
        # one fresh generator per optimizer instance it creates, all derived from run.seed.
        self.rng = stream("engine")
        self._n_optimizers = 0
        self.algo_evals = {'GA': 0, 'PSO': 0, 'DE': 0, 'CMAES': 0}

        self.evals_used = 0
        self.best_individual = None
        self.history = []
        self.evaluated = []          # every evaluated Individual (confirm-before-select, Phase 3.7)
        # Phase 4.1: what the controller measured and decided, for the ablation and the summary
        self.scoring_history = []    # one entry per scoring call: evals used + every island's metrics
        self.decisions = []          # promotions, swaps, rollbacks, injections, elite sharing
        self.cooldowns = {'GA': 0, 'PSO': 0, 'DE': 0, 'CMAES': 0}
        self.active_grace_period = None
        # One CSV row per evaluation (population.csv of the run folder), if a path is given
        self.pop_logger = PopulationLogger(self.space, population_csv) if population_csv else None
        self.reporter = get_reporter()
        # Per family: best so far, best at the last scoring call, evaluation seconds since the
        # last scoring call (efficiency), and island bests at past scoring calls (trend).
        self._algo_stats = {k: {'best': 0.0, 'prev_best': 0.0, 'eval_seconds': 0.0, 'best_history': []}
                            for k in ('GA', 'PSO', 'DE', 'CMAES')}

        island = int(self.pop_size * self.t1_fraction) // 4
        if island < self.min_island_size:
            raise ValueError(
                f"pop_size={self.pop_size} with t1_fraction={self.t1_fraction} gives Tier 1 islands of {island}; "
                f"at least engine.min_island_size={self.min_island_size} are needed (raise pop_size)")

    def _new_optimizer(self, algo_key, pop_size):
        """Create an optimizer with its own generator: stream("engine", "optimizer", algo, n)."""
        self._n_optimizers += 1
        return _make_optimizer(algo_key, self.space, pop_size,
                               rng=stream("engine", "optimizer", algo_key, self._n_optimizers))

    def _evaluate_batch(self, vectors, algo_name):
        algo_key = algo_name.replace("T2_", "").replace("T1_", "")
        inds = []
        rem = self.budget - self.evals_used
        if rem <= 0:
            return inds
        eval_vectors = vectors[:rem]
        if not eval_vectors:
            return inds

        def _eval_single(vec):
            ind = Individual(vec)
            fitness = self.eval.evaluate(ind)
            ind.set_fitness(fitness)
            ind.source_algorithm = algo_name
            return ind

        n_workers = min(self.max_workers, len(eval_vectors))
        t_start = time.perf_counter()
        if n_workers > 1:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                evaluated_inds = list(executor.map(_eval_single, eval_vectors))
        else:
            evaluated_inds = [_eval_single(v) for v in eval_vectors]
        if algo_key in self._algo_stats:
            self._algo_stats[algo_key]['eval_seconds'] += time.perf_counter() - t_start

        for ind in evaluated_inds:
            inds.append(ind)
            self.evaluated.append(ind)
            self.evals_used += 1
            if algo_key in self.algo_evals:
                self.algo_evals[algo_key] += 1
            
            # Log individual to CSV
            if self.pop_logger is not None:
                self.pop_logger.log(self.evals_used, algo_name, ind)
            
            fitness = ind.fitness
            if algo_key in self._algo_stats:
                if fitness > self._algo_stats[algo_key]['best']:
                    self._algo_stats[algo_key]['best'] = fitness
                    self.reporter.emit("metric", {"algo": algo_key, "accuracy": fitness})
            
            if self.best_individual is None or fitness > self.best_individual.fitness:
                self.best_individual = ind

            self.history.append({
                'eval_id': self.evals_used,
                'fitness': fitness,
                'best_so_far': self.best_individual.fitness if self.best_individual else fitness,
                'algo': algo_name
            })
        return inds

    @staticmethod
    def _latent_diversity(pop, dims):
        """Mean pairwise Euclidean distance between the island's latent vectors / sqrt(d), in [0, 1]."""
        if len(pop) < 2:
            return 0.0
        X = np.array([ind.vector for ind in pop], dtype=float)
        diff = X[:, None, :] - X[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))
        n = len(X)
        return float(dist[np.triu_indices(n, k=1)].mean() / np.sqrt(dims))

    def _trend(self, history):
        """Least-squares slope of the island's best fitness over the last `trend_window` scoring calls."""
        h = history[-self.trend_window:]
        if len(h) < 2:
            return 0.0
        return float(np.polyfit(np.arange(len(h)), h, 1)[0])

    def _compute_composite(self, algo_pops):
        """
        Raw metrics per island, composite score (CompositeScoringEngine) and UCB score (UCB1Bandit).
          fitness     best accuracy in the island
          speed       improvement of the family's best since its last scoring call
          efficiency  that improvement per second of evaluation time spent on the family since then
          diversity   mean pairwise latent distance / sqrt(d)
          trend       slope of the family's best over its last `trend_window` scoring calls
        Returns (ucb_scores, raw): raw[name] also holds 'raw_composite' and 'ucb_bonus'.
        """
        dims = len(list(self.space.keys()))
        algo_data = {}
        for name, pop in algo_pops.items():
            if not pop: continue
            best = max(ind.fitness for ind in pop)
            algo_key = name.replace("T1_", "").replace("T2_", "")
            st = self._algo_stats[algo_key]
            improvement = max(best - st['prev_best'], 0.0)
            efficiency = improvement / max(st['eval_seconds'], 1e-6)
            st['best_history'].append(best)
            algo_data[name] = {
                'fitness': best,
                'speed': improvement,
                'efficiency': efficiency,
                'diversity': self._latent_diversity(pop, dims),
                'trend': self._trend(st['best_history']),
                'improvement': improvement,
            }
            st['prev_best'] = best
            st['eval_seconds'] = 0.0

        composite_scores = self.composite_engine.calculate_scores(algo_data)

        ucb_scores = {}
        for name, comp in composite_scores.items():
            algo_key = name.replace("T1_", "").replace("T2_", "")
            n_algo = self.algo_evals.get(algo_key, 0)
            self.bandit.update(algo_key, comp)
            ucb_scores[name] = self.bandit.score(algo_key, comp, n_algo, self.evals_used)
            algo_data[name]['raw_composite'] = comp
            algo_data[name]['ucb_bonus'] = self.bandit.bonus(n_algo, self.evals_used)
        self.scoring_history.append({
            "evals": self.evals_used,
            "islands": {name: {"fitness": d["fitness"], "speed": d["speed"], "efficiency": d["efficiency"],
                               "diversity": d["diversity"], "trend": d["trend"],
                               "composite": d["raw_composite"], "ucb": ucb_scores[name]}
                        for name, d in algo_data.items()},
        })
        return ucb_scores, algo_data

    def _switch(self, details):
        """Emit a 'switch' event to the dashboard and record it in self.decisions."""
        self.decisions.append({"evals": self.evals_used, **details})
        self.reporter.emit("switch", details)

    def _print_scoring_table(self, algo_pops, scores, raw):
        print(f"\n  {'Algorithm':<12} {'Best Acc':>9} {'Diversity':>10} {'Composite':>10} {'Bonus':>7} {'UCB Score':>10}")
        print(f"  {'-'*12} {'-'*9} {'-'*10} {'-'*10} {'-'*7} {'-'*10}")
        sorted_algos = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        for name in sorted_algos:
            print(f"  {name:<12} {raw[name]['fitness']:>8.2f}%"
                  f" {raw[name]['diversity']:>10.3f}"
                  f" {raw[name].get('raw_composite', 0.0):>10.3f}"
                  f" {raw[name].get('ucb_bonus', 0.0):>7.3f}"
                  f" {scores.get(name, 0.0):>10.3f}")
            self.reporter.emit("composite", {"algo": name, "composite": scores.get(name, 0.0), "diversity": raw[name]['diversity']})

    def _random_inject(self, optimizer, inject_pct, dims, tag):
        n_inject = max(1, int(len(optimizer.population) * inject_pct))
        print(f"\n  [INJECT] Replacing bottom {n_inject} in {tag} with fresh random vectors.")
        self._switch({"action": "Injection", "algo": tag, "details": f"Replaced bottom {n_inject} due to stagnation"})
        fresh = self._evaluate_batch([self.rng.random(dims) for _ in range(n_inject)], tag)
        optimizer.inject(fresh)     # Phase 1.8: the optimizer updates its own search state too
        return optimizer

    def _initialize_population(self, dims):
        """Step 1: evaluate pop_size random vectors in parallel chunks; returns them best first."""
        self.reporter.emit("step", {"num": 1, "name": f"Initial Population (N={self.pop_size})"})
        print(f"\n{_sep()}")
        print(_header(f"STEP 1: INITIAL POPULATION (N={self.pop_size})"))
        print(_sep())
        full_pop = []
        chunk_size = max(1, self.max_workers * 2)
        for batch_start in range(0, self.pop_size, chunk_size):
            if self.evals_used >= self.budget: break
            curr_len = min(chunk_size, self.pop_size - batch_start)
            batch_inds = self._evaluate_batch([self.rng.random(dims) for _ in range(curr_len)], 'Init')
            full_pop.extend(batch_inds)
            print(f"    Evaluated {len(full_pop)}/{self.pop_size} (best so far: {max(ind.fitness for ind in full_pop):.2f}%)")
            self.reporter.emit("progress", {"evals": self.evals_used, "budget": self.budget})
        full_pop.sort(key=lambda x: x.fitness, reverse=True)
        print(f"  [OK] Full population evaluated. Best: {full_pop[0].fitness:.2f}% | Budget: {self.evals_used}/{self.budget}")
        return full_pop

    def _set_checkpoint_interval(self, island_size):
        """
        Phase 1.7: with auto_k, shrink k so Tier 2 reaches at least min_t2_checkpoints checkpoints.
        Estimate: Tier 1 uses t1_iters · 4 · island evaluations after the initial population; each
        Tier 2 iteration evaluates about pop_size candidates (both T2 islands + both T1 islands).
        k_iters stays the upper bound.
        """
        self._k_note = ""
        if not self.auto_k:
            return
        remaining = self.budget - self.pop_size - self.t1_iters * 4 * island_size
        t2_iters = max(0, remaining // max(1, self.pop_size))
        configured = self.k
        if t2_iters >= self.min_t2_checkpoints:
            self.k = max(1, min(configured, t2_iters // self.min_t2_checkpoints))
        else:
            self.k = 1
            print(f"  [WARNING] budget {self.budget} leaves about {t2_iters} Tier 2 iteration(s); fewer than "
                  f"{self.min_t2_checkpoints} checkpoints are possible. Raise the budget or lower pop_size / t1_iters.")
        self._k_note = (f" (auto: about {t2_iters} Tier 2 iterations, >= {self.min_t2_checkpoints} checkpoints"
                        + (f"; k_iters={configured} lowered" if self.k < configured else "") + ")")

    def run(self):
        total_start = time.time()
        dims = len(list(self.space.keys()))

        # ==============================================================
        # STEP 0: Search Space
        # ==============================================================
        self.reporter.emit("step", {"num": 0, "name": "Search Space Definition"})
        print(_sep())
        print(_header("STEP 0: SEARCH SPACE DEFINITION"))
        print(_sep())
        print(f"  Hyperparameter Space : {dims} dimensions")
        print(f"  Population Size      : {self.pop_size}")
        print(f"  Budget               : {self.budget} evals")
        print(f"  T1 Competition       : {self.t1_iters} iters (4 Algorithms)")
        print(f"  Tier split           : {self.tier_split}")

        # ==============================================================
        # STEP 1: INITIAL POPULATION
        # ==============================================================
        full_pop = self._initialize_population(dims)

        # ==============================================================
        # STEP 2: TIER SPLIT (TierManager: Tier 1 islands + Tier 2 reserve)
        # ==============================================================
        self.reporter.emit("step", {"num": 2, "name": "Tier Split (4-Way)"})
        t1_island_size = int(self.pop_size * self.t1_fraction) // 4
        tiers = TierManager(full_pop, t1_island_size, n_islands=4, split=self.tier_split,
                            rng=stream("engine", "tiers"))
        seeds = dict(zip(['GA', 'PSO', 'DE', 'CMAES'], tiers.islands()))
        t2_reserve_groups = tiers.reserve_groups(self.n_promoted)
        t1_count = 4 * t1_island_size
        self._set_checkpoint_interval(t1_island_size)

        print(f"\n{_sep()}")
        print(_header("STEP 2: TIER SPLIT (4-WAY)"))
        print(_sep())
        print(f"  Split method : {self.tier_split}")
        print(f"  Tier 1       : {t1_count} individuals split 4 ways ({t1_island_size} each)")
        print(f"  Tier 2       : {len(tiers.tier2_pool)} individuals RESERVED for the top {self.n_promoted}")
        print(f"  Checkpoint k : {self.k}{self._k_note}")

        self.reporter.emit("tier_state", {"T1": ["GA", "PSO", "DE", "CMAES"], "T2": ["(Reserved)", "(Reserved)"]})

        # ==============================================================
        # STEP 3: TIER 1 COMPETITION
        # ==============================================================
        self.reporter.emit("step", {"num": 3, "name": f"Tier 1 Competition ({self.t1_iters} iters)"})
        print(f"\n{_sep()}")
        print(_header(f"STEP 3: TIER 1 COMPETITION ({self.t1_iters} iters)"))
        print(_sep())

        t1_optims = {k: self._new_optimizer(k, t1_island_size) for k in seeds}
        for k, opt in t1_optims.items():
            opt.tell(seeds[k])

        print(f"  {'Iter':>4}  {'GA':>8}  {'PSO':>8}  {'DE':>8}  {'CMAES':>8}  {'Global':>8}  {'Time':>5}  {'Budget':>11}")
        print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*5}  {'-'*11}")

        for i in range(1, self.t1_iters + 1):
            if self.evals_used >= self.budget: break
            ts = time.time()
            prev = self.best_individual.fitness if self.best_individual else 0.0

            for k, opt in t1_optims.items():
                inds = self._evaluate_batch(opt.ask(t1_island_size), f"T1_{k}")
                if inds: opt.tell(inds)

            bests = {k: max((ind.fitness for ind in opt.population), default=0.0) for k, opt in t1_optims.items()}
            glob = self.best_individual.fitness if self.best_individual else 0.0
            flag = " <-- NEW BEST" if glob > prev else ""
            
            print(f"  {i:>4}  {bests['GA']:>7.2f}%  {bests['PSO']:>7.2f}%  {bests['DE']:>7.2f}%  {bests['CMAES']:>7.2f}%  {glob:>7.2f}%  {time.time()-ts:>4.1f}s  {self.evals_used:>4}/{self.budget}{flag}")
            self.reporter.emit("progress", {"evals": self.evals_used, "budget": self.budget})
            
            # Tier 1 Stagnation Check
            if i % self.k == 0:
                t1_pops = {f"T1_{k}": list(opt.population) for k, opt in t1_optims.items()}
                _, rw = self._compute_composite(t1_pops)
                for k, opt in t1_optims.items():
                    if rw.get(f"T1_{k}", {}).get('improvement', 1.0) == 0.0:
                        t1_optims[k] = self._random_inject(opt, self.t1_inject_pct, dims, f"T1_{k}")

        # ==============================================================
        # STEP 4: PROMOTION (TOP 2 to T2)
        # ==============================================================
        self.reporter.emit("step", {"num": 4, "name": "Composite Scoring + Top-2 Promotion"})
        print(f"\n{_sep()}")
        print(_header("STEP 4: COMPOSITE SCORING + TOP-2 PROMOTION"))
        print(_sep())
        
        t1_pops = {f"T1_{k}": list(opt.population) for k, opt in t1_optims.items()}
        scores, raw = self._compute_composite(t1_pops)
        self._print_scoring_table(t1_pops, scores, raw)

        sorted_algos = self.bandit.rank(scores)          # Phase 1.1: ranked by the bandit's UCB score
        top2 = sorted_algos[:self.n_promoted]
        bot2 = sorted_algos[self.n_promoted:]

        print(f"\n  [PROMOTED to Tier 2] : {top2[0]}, {top2[1]}")
        print(f"  [RETAINED in Tier 1] : {bot2[0]}, {bot2[1]}")
        
        t2_clean = [t.replace("T1_","") for t in top2]
        t1_clean = [t.replace("T1_","") for t in bot2]
        self.reporter.emit("tier_state", {"T1": t1_clean, "T2": t2_clean})
        self._switch({"action": "Promotion", "algo": "System", "details": f"Promoted {t2_clean} to Tier 2"})

        # Build T2 Optimizers
        t2_optims = {}
        for idx, t1_name in enumerate(top2):
            k = t1_name.replace("T1_", "")
            seed_pop = t1_pops[t1_name] + t2_reserve_groups[idx]
            opt = self._new_optimizer(k, len(seed_pop))
            opt.tell(seed_pop)
            t2_optims[k] = opt

        # Retain T1 Optimizers
        active_t1_optims = {name.replace("T1_", ""): t1_optims[name.replace("T1_", "")] for name in bot2}
        t1_pop_size = t1_island_size
        stagnation_counters = {k: 0 for k in t2_optims}

        # ==============================================================
        # STEP 5: TIER 2 WORKING LOOP
        # ==============================================================
        def _print_t2_header():
            t2_list = list(t2_optims.keys())
            t1_list = list(active_t1_optims.keys())
            self.reporter.emit("step", {"num": 5, "name": f"Tier 2 {t2_list} + Tier 1 {t1_list}"})
            print(f"\n{_sep()}")
            print(_header(f"STEP 5: TIER 2 {t2_list} + TIER 1 {t1_list}"))
            print(_sep())
            
            algos = [f"T2_{k}" for k in t2_optims.keys()] + [f"T1_{k}" for k in active_t1_optims.keys()]
            header_str = "  ".join(f"{a:>8}" for a in algos)
            print(f"  {'Iter':>4}  {header_str}  {'Global':>8}  {'Time':>5}  {'Budget':>11}")
            print(f"  {'-'*4}  " + "  ".join("-" * 8 for _ in algos) + f"  {'-'*8}  {'-'*5}  {'-'*11}")
            return algos

        curr_layout = _print_t2_header()

        
        iteration = 0
        while self.evals_used < self.budget:
            iteration += 1
            for k in self.cooldowns:
                if self.cooldowns[k] > 0: self.cooldowns[k] -= 1
                
            prev_best = self.best_individual.fitness if self.best_individual else 0.0

            ts = time.time()
            for k, opt in t2_optims.items():
                inds = self._evaluate_batch(opt.ask(opt.pop_size), f"T2_{k}")
                if inds: opt.tell(inds)

            for k, opt in active_t1_optims.items():
                inds = self._evaluate_batch(opt.ask(opt.pop_size), f"T1_{k}")
                if inds: opt.tell(inds)

            if self.evals_used >= self.budget: break

            glob = self.best_individual.fitness if self.best_individual else 0.0
            new_b = glob > prev_best
            flag = " <-- NEW BEST" if new_b else ""
            
            bests = {}
            for k, opt in t2_optims.items(): bests[f"T2_{k}"] = max((i.fitness for i in opt.population), default=0.0)
            for k, opt in active_t1_optims.items(): bests[f"T1_{k}"] = max((i.fitness for i in opt.population), default=0.0)
            
            scores_str = "  ".join(f"{bests[a]:>7.2f}%" for a in curr_layout)
            print(f"  {iteration:>4}  {scores_str}  {glob:>7.2f}%  {time.time()-ts:>4.1f}s  {self.evals_used:>4}/{self.budget}{flag}")
            
            self.reporter.emit("progress", {"evals": self.evals_used, "budget": self.budget})
            
            if new_b:
                # User requested to remove the hyperparameter printout here
                pass

            if self.active_grace_period:
                self.active_grace_period['iters_left'] -= 1
                if glob > self.active_grace_period['baseline']:
                    self.active_grace_period = None # Success!
                elif self.active_grace_period['iters_left'] <= 0:
                    p_algo = self.active_grace_period['algo']
                    d_algo = self.active_grace_period['target_t1']
                    print(f"\n  [ROLLBACK] {p_algo} failed to improve global best. Rolling back to {d_algo}.")
                    self._switch({"action": "Rollback", "algo": f"{p_algo} -> {d_algo}", "details": "Promoted algorithm failed grace period"})
                    
                    t2_optims[d_algo] = self.active_grace_period['snapshot']
                    del t2_optims[p_algo]
                    
                    new_t1 = self._new_optimizer(p_algo, t1_pop_size)
                    new_t1.tell(self._evaluate_batch([self.rng.random(dims) for _ in range(t1_pop_size)], f"T1_{p_algo}"))
                    active_t1_optims[p_algo] = new_t1
                    if d_algo in active_t1_optims: del active_t1_optims[d_algo]
                    
                    self.active_grace_period = None
                    self.cooldowns[p_algo] = self.cooldown_multiplier * self.k # Penalty
                    self.reporter.emit("tier_state", {"T1": list(active_t1_optims.keys()), "T2": list(t2_optims.keys())})
                    
                    curr_layout = _print_t2_header()

            if iteration % self.k == 0:
                print(f"\n  {_sep('-', 80)}")
                print(f"  {_header('SWITCHING CHECKPOINT', '-', 80)}")
                
                all_pops = {}
                for k, opt in t2_optims.items(): all_pops[f"T2_{k}"] = list(opt.population)
                for k, opt in active_t1_optims.items(): all_pops[f"T1_{k}"] = list(opt.population)
                
                sc, rw = self._compute_composite(all_pops)
                self._print_scoring_table(all_pops, sc, rw)

                t1_scores = {k: sc[k] for k in sc if k.startswith("T1_")}
                t2_scores = {k: sc[k] for k in sc if k.startswith("T2_")}
                
                best_t1_name = self.bandit.rank(t1_scores)[0]
                worst_t2_name = self.bandit.rank(t2_scores)[-1]
                best_t1_key = best_t1_name.replace("T1_", "")
                worst_t2_key = worst_t2_name.replace("T2_", "")

                # a) Swap
                if t1_scores[best_t1_name] > t2_scores[worst_t2_name] + self.swap_margin and self.cooldowns[best_t1_key] == 0 and self.active_grace_period is None:
                    print(f"\n  [SWAP] {best_t1_name} (UCB {t1_scores[best_t1_name]:.3f}) beats {worst_t2_name} "
                          f"(UCB {t2_scores[worst_t2_name]:.3f}) by more than the margin {self.swap_margin:.2f}")
                    self._switch({"action": "Swap", "algo": f"{best_t1_key} <-> {worst_t2_key}", "details": "Demoting T2 algorithm due to low composite score"})
                    
                    import copy
                    self.active_grace_period = {
                        'algo': best_t1_key,
                        'iters_left': self.grace_iters,
                        'baseline': glob,
                        'snapshot': copy.deepcopy(t2_optims[worst_t2_key]),
                        'target_t1': worst_t2_key
                    }
                    self.cooldowns[worst_t2_key] = self.cooldown_multiplier * self.k
                    
                    old_t2 = list(t2_optims[worst_t2_key].population)
                    old_t1 = list(active_t1_optims[best_t1_key].population)
                    t2_size_target = t2_optims[worst_t2_key].pop_size
                    
                    new_t2 = self._new_optimizer(best_t1_key, t2_size_target)
                    new_t2.tell(old_t1 + old_t2[len(old_t1):])
                    
                    new_t1 = self._new_optimizer(worst_t2_key, t1_pop_size)
                    new_t1.tell(old_t2[:t1_pop_size])
                    
                    t2_optims[best_t1_key] = new_t2
                    del t2_optims[worst_t2_key]
                    
                    active_t1_optims[worst_t2_key] = new_t1
                    del active_t1_optims[best_t1_key]
                    
                    stagnation_counters[best_t1_key] = 0
                    if worst_t2_key in stagnation_counters: del stagnation_counters[worst_t2_key]
                    
                    self.reporter.emit("tier_state", {"T1": list(active_t1_optims.keys()), "T2": list(t2_optims.keys())})

                # b) Stagnation
                for k, opt in list(t2_optims.items()):
                    if rw.get(f"T2_{k}", {}).get('improvement', 1.0) == 0.0:
                        stagnation_counters[k] = stagnation_counters.get(k, 0) + 1
                        if stagnation_counters[k] >= self.stagnation_limit:
                            t2_optims[k] = self._random_inject(opt, self.inject_pct, dims, f"T2_{k}")
                            stagnation_counters[k] = 0
                    else:
                        stagnation_counters[k] = 0

                # c) Elite Sharing (Tier 2 ONLY)
                # Phase 1.9: pairwise sharing (TierManager): each island's worst are replaced by the
                # other island's elites, through optimizer.inject() so search state follows.
                (ka, oa), (kb, ob) = list(t2_optims.items())[:2]
                n_share = max(1, int(min(len(oa.population), len(ob.population)) * self.elite_pct))
                elites_a = TierManager.select_elites(oa.population, n_share / max(1, len(oa.population)))[:n_share]
                elites_b = TierManager.select_elites(ob.population, n_share / max(1, len(ob.population)))[:n_share]
                print(f"\n  [ELITE SHARE] T2_{ka} <-> T2_{kb}: exchanging the top {n_share} "
                      f"({int(self.elite_pct*100)}%) of each island")
                self._switch({"action": "Elite Sharing", "algo": f"{ka} <-> {kb}",
                                              "details": f"Pairwise exchange of the top {n_share} between T2 islands"})
                oa.inject(elites_b)
                ob.inject(elites_a)
                
                print(f"  {_sep('-', 80)}\n")
                
                # Reprint the header cleanly after all checkpoint outputs are finished
                curr_layout = _print_t2_header()

        # ==============================================================
        # FINAL RESULTS
        # ==============================================================
        self.reporter.emit("step", {"num": 6, "name": "Final Results"})
        print(f"\n{_sep()}")
        print(_header("FINAL RESULTS"))
        print(_sep())
        print(f"  Total Evaluations  : {self.evals_used}")
        
        total_time = time.time()-total_start
        mins = int(total_time // 60)
        secs = total_time % 60
        print(f"  Total Time         : {mins}m {secs:.1f}s")
        
        print(f"  Winning Algorithm  : {self.best_individual.source_algorithm}")
        print(f"  Best Accuracy      : {self.best_individual.fitness:.4f}%")
        params = _decode_individual(self.best_individual, self.space)
        if "model_type" in params:
            print(f"  Optimal Model Type : {params['model_type']}")
        print(f"\n  Best Hyperparameters:")
        for k, v in params.items():
            print(f"    * {k:<20}: {v:.4f}" if isinstance(v, float) else f"    * {k:<20}: {v}")
        print(_sep())

        if self.pop_logger is not None:
            self.pop_logger.close()
        return self.best_individual
