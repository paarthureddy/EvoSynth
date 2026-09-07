import numpy as np
import time
import sys
import threading
import requests
from automl_engine.engine.tier_manager import TierManager
from automl_engine.engine.composite import CompositeScoringEngine
from automl_engine.engine.ucb1 import UCB1Bandit
from automl_engine.optimizers.ga import GeneticAlgorithm
from automl_engine.optimizers.pso import ParticleSwarm
from automl_engine.optimizers.de import DifferentialEvolution
from automl_engine.optimizers.cmaes import CMAESOptimizer
from automl_engine.core.individual import Individual

# ------------------------------------------------------------------
# UI / Dashboard Reporter
# ------------------------------------------------------------------
class EventReporter:
    def __init__(self):
        self.url = "http://localhost:8000/event"
        self.enabled = False
        try:
            # Check if API server is running
            requests.get("http://localhost:8000/docs", timeout=0.5)
            self.enabled = True
        except:
            pass

    def emit(self, event_type, payload):
        if not self.enabled: return
        def _send():
            try:
                requests.post(self.url, json={"type": event_type, "data": payload}, timeout=1)
            except:
                pass
        threading.Thread(target=_send, daemon=True).start()

reporter = EventReporter()

class LoggerWriter:
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.buffer = ""
        # Open a file to mirror all terminal output live
        self.log_file = open("run_output.txt", "w", encoding="utf-8")

    def write(self, message):
        self.original_stdout.write(message)
        self.original_stdout.flush()
        
        # Write to file and flush immediately to ensure it's saved on crash/cancel
        self.log_file.write(message)
        self.log_file.flush()
        
        # Only emit non-empty lines to reduce network spam
        if message and message != '\n':
            reporter.emit("log", {"text": message})

    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()

# Intercept prints to stream them live
sys.stdout = LoggerWriter(sys.stdout)

class PopulationLogger:
    def __init__(self, space):
        self.space = space
        self.files = {
            'Init': open('warmup_population.csv', 'w', encoding='utf-8'),
            'T1': open('tier1_population.csv', 'w', encoding='utf-8'),
            'T2': open('tier2_population.csv', 'w', encoding='utf-8')
        }
        
        # Write headers based on search space
        hp_names = list(space.keys())
        header = "Eval_ID,Algorithm,Fitness," + ",".join(hp_names) + "\n"
        for f in self.files.values():
            f.write(header)
            f.flush()

    def log(self, eval_id, algo_name, ind):
        if algo_name == 'Init':
            f = self.files['Init']
        elif algo_name.startswith('T1'):
            f = self.files['T1']
        elif algo_name.startswith('T2'):
            f = self.files['T2']
        else:
            return
            
        params = _decode_individual(ind, self.space)
        row = f"{eval_id},{algo_name},{ind.fitness:.4f}," + ",".join(str(params[k]) for k in self.space.keys()) + "\n"
        f.write(row)
        f.flush()

# ------------------------------------------------------------------

def _sep(char="=", width=80):
    return char * width

def _header(title, char="=", width=80):
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
    if algo_key == 'GA': return GeneticAlgorithm(space, pop_size)
    if algo_key == 'PSO': return ParticleSwarm(space, pop_size)
    if algo_key == 'DE': return DifferentialEvolution(space, pop_size)
    if algo_key == 'CMAES': return CMAESOptimizer(space, pop_size)
    raise ValueError(f"Unknown algo: {algo_key}")

class DynamicOptimizer:
    def __init__(self, search_space, evaluator,
                 budget=2000, pop_size=100,
                 t1_iters=30, k_iters=10,
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

        self.c = 0.1
        self.algo_evals = {'GA': 0, 'PSO': 0, 'DE': 0, 'CMAES': 0}

        self.evals_used = 0
        self.best_individual = None
        self.cooldowns = {'GA': 0, 'PSO': 0, 'DE': 0, 'CMAES': 0}
        self.active_grace_period = None
        self.pop_logger = PopulationLogger(self.space)
        self._algo_stats = {
            'GA': {'best': 0.0, 'prev_best': 0.0},
            'PSO': {'best': 0.0, 'prev_best': 0.0},
            'DE': {'best': 0.0, 'prev_best': 0.0},
            'CMAES': {'best': 0.0, 'prev_best': 0.0}
        }

    def _evaluate_batch(self, vectors, algo_name):
        algo_key = algo_name.replace("T2_", "").replace("T1_", "")
        inds = []
        for vec in vectors:
            if self.evals_used >= self.budget: break
            ind = Individual(vec)
            fitness = self.eval.evaluate(ind)
            ind.set_fitness(fitness)
            ind.source_algorithm = algo_name
            inds.append(ind)
            self.evals_used += 1
            if algo_key in self.algo_evals: self.algo_evals[algo_key] += 1
            
            # Log individual to CSV
            self.pop_logger.log(self.evals_used, algo_name, ind)
            
            if algo_key in self._algo_stats:
                if fitness > self._algo_stats[algo_key]['best']:
                    self._algo_stats[algo_key]['best'] = fitness
                    reporter.emit("metric", {"algo": algo_key, "accuracy": fitness})
            
            if self.best_individual is None or fitness > self.best_individual.fitness:
                self.best_individual = ind
        return inds

    def _compute_composite(self, algo_pops):
        algo_data = {}
        for name, pop in algo_pops.items():
            if not pop: continue
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
            
        composite_scores = self.composite_engine.calculate_scores(algo_data)
        
        final_scores = {}
        import math
        for name, comp_score in composite_scores.items():
            algo_key = name.replace("T1_", "").replace("T2_", "")
            algo_budget = self.algo_evals.get(algo_key, 1)
            total_budget = max(1, self.evals_used)
            
            exploration_bonus = self.c * math.sqrt(math.log(total_budget) / max(1, algo_budget))
            final_scores[name] = comp_score + exploration_bonus
            algo_data[name]['ucb_bonus'] = exploration_bonus
            algo_data[name]['raw_composite'] = comp_score
            
        return final_scores, algo_data

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
            reporter.emit("composite", {"algo": name, "composite": scores.get(name, 0.0), "diversity": raw[name]['diversity']})

    def _random_inject(self, optimizer, inject_pct, dims, tag):
        n_inject = max(1, int(len(optimizer.population) * inject_pct))
        print(f"\n  [INJECT] Replacing bottom {n_inject} in {tag} with fresh random vectors.")
        reporter.emit("switch", {"action": "Injection", "algo": tag, "details": f"Replaced bottom {n_inject} due to stagnation"})
        fresh = self._evaluate_batch([np.random.rand(dims) for _ in range(n_inject)], tag)
        optimizer.population.sort(key=lambda x: x.fitness, reverse=True)
        optimizer.population = optimizer.population[:-n_inject] + fresh
        return optimizer

    def run(self):
        total_start = time.time()
        dims = len(list(self.space.keys()))

        # ==============================================================
        # STEP 0: Search Space
        # ==============================================================
        reporter.emit("step", {"num": 0, "name": "Search Space Definition"})
        print(_sep())
        print(_header("STEP 0: SEARCH SPACE DEFINITION"))
        print(_sep())
        print(f"  Hyperparameter Space : {dims} dimensions")
        print(f"  Population Size      : {self.pop_size}")
        print(f"  Budget               : {self.budget} evals")
        print(f"  T1 Competition       : {self.t1_iters} iters (4 Algorithms)")

        # ==============================================================
        # STEP 1: INITIAL POPULATION
        # ==============================================================
        reporter.emit("step", {"num": 1, "name": "Initial Population (N=100)"})
        print(f"\n{_sep()}")
        print(_header(f"STEP 1: INITIAL POPULATION (N={self.pop_size})"))
        print(_sep())
        full_pop = []
        for i in range(self.pop_size):
            if self.evals_used >= self.budget: break
            full_pop.extend(self._evaluate_batch([np.random.rand(dims)], 'Init'))
            if (i + 1) % 20 == 0:
                print(f"    Evaluated {i+1}/{self.pop_size} (best so far: {max(ind.fitness for ind in full_pop):.2f}%)")
                reporter.emit("progress", {"evals": self.evals_used, "budget": self.budget})
        
        full_pop.sort(key=lambda x: x.fitness, reverse=True)
        print(f"  [OK] Full population evaluated. Best: {full_pop[0].fitness:.2f}% | Budget: {self.evals_used}/{self.budget}")

        # ==============================================================
        # STEP 2: TIER SPLIT (40% T1 / 60% T2 Reserved)
        # ==============================================================
        reporter.emit("step", {"num": 2, "name": "Tier Split (4-Way)"})
        t1_count = int(self.pop_size * 0.4)
        t2_reserved = full_pop[t1_count:]
        t1_island_size = t1_count // 4

        seeds = {
            'GA': full_pop[:t1_island_size],
            'PSO': full_pop[t1_island_size:2*t1_island_size],
            'DE': full_pop[2*t1_island_size:3*t1_island_size],
            'CMAES': full_pop[3*t1_island_size:t1_count]
        }

        print(f"\n{_sep()}")
        print(_header("STEP 2: TIER SPLIT (4-WAY)"))
        print(_sep())
        print(f"  Tier 1 (40%) : {t1_count} individuals split 4 ways ({t1_island_size} each)")
        print(f"  Tier 2 (60%) : {len(t2_reserved)} individuals RESERVED for Top 2 winners")
        
        reporter.emit("tier_state", {"T1": ["GA", "PSO", "DE", "CMAES"], "T2": ["(Reserved)", "(Reserved)"]})

        # ==============================================================
        # STEP 3: TIER 1 COMPETITION
        # ==============================================================
        reporter.emit("step", {"num": 3, "name": f"Tier 1 Competition ({self.t1_iters} iters)"})
        print(f"\n{_sep()}")
        print(_header(f"STEP 3: TIER 1 COMPETITION ({self.t1_iters} iters)"))
        print(_sep())

        t1_optims = {k: _make_optimizer(k, self.space, t1_island_size) for k in seeds}
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
            reporter.emit("progress", {"evals": self.evals_used, "budget": self.budget})
            
            # Tier 1 Stagnation Check
            if i % self.k == 0:
                t1_pops = {f"T1_{k}": list(opt.population) for k, opt in t1_optims.items()}
                _, rw = self._compute_composite(t1_pops)
                for k, opt in t1_optims.items():
                    if rw.get(f"T1_{k}", {}).get('trend', 1.0) == 0.0:
                        t1_optims[k] = self._random_inject(opt, 0.50, dims, f"T1_{k}")

        # ==============================================================
        # STEP 4: PROMOTION (TOP 2 to T2)
        # ==============================================================
        reporter.emit("step", {"num": 4, "name": "Composite Scoring + Top-2 Promotion"})
        print(f"\n{_sep()}")
        print(_header("STEP 4: COMPOSITE SCORING + TOP-2 PROMOTION"))
        print(_sep())
        
        t1_pops = {f"T1_{k}": list(opt.population) for k, opt in t1_optims.items()}
        scores, raw = self._compute_composite(t1_pops)
        self._print_scoring_table(t1_pops, scores, raw)

        sorted_algos = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        top2 = sorted_algos[:2]
        bot2 = sorted_algos[2:]

        for k in scores:
            algo_key = k.replace("T1_", "")
            self.bandit.register_algorithm(algo_key)
            self.bandit.update(algo_key, scores[k])

        print(f"\n  [PROMOTED to Tier 2] : {top2[0]}, {top2[1]}")
        print(f"  [RETAINED in Tier 1] : {bot2[0]}, {bot2[1]}")
        
        t2_clean = [t.replace("T1_","") for t in top2]
        t1_clean = [t.replace("T1_","") for t in bot2]
        reporter.emit("tier_state", {"T1": t1_clean, "T2": t2_clean})
        reporter.emit("switch", {"action": "Promotion", "algo": "System", "details": f"Promoted {t2_clean} to Tier 2"})

        # Build T2 Optimizers
        t2_optims = {}
        t2_ind_per_algo = len(t2_reserved) // 2
        for idx, t1_name in enumerate(top2):
            k = t1_name.replace("T1_", "")
            t2_size = len(t1_pops[t1_name]) + t2_ind_per_algo
            opt = _make_optimizer(k, self.space, t2_size)
            seed_pop = t1_pops[t1_name] + t2_reserved[idx*t2_ind_per_algo : (idx+1)*t2_ind_per_algo]
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
            reporter.emit("step", {"num": 5, "name": f"Tier 2 {t2_list} + Tier 1 {t1_list}"})
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
            
            reporter.emit("progress", {"evals": self.evals_used, "budget": self.budget})
            
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
                    reporter.emit("switch", {"action": "Rollback", "algo": f"{p_algo} -> {d_algo}", "details": "Promoted algorithm failed grace period"})
                    
                    t2_optims[d_algo] = self.active_grace_period['snapshot']
                    del t2_optims[p_algo]
                    
                    new_t1 = _make_optimizer(p_algo, self.space, t1_pop_size)
                    new_t1.tell(self._evaluate_batch([np.random.rand(dims) for _ in range(t1_pop_size)], f"T1_{p_algo}"))
                    active_t1_optims[p_algo] = new_t1
                    if d_algo in active_t1_optims: del active_t1_optims[d_algo]
                    
                    self.active_grace_period = None
                    self.cooldowns[p_algo] = 2 * self.k # Penalty
                    reporter.emit("tier_state", {"T1": list(active_t1_optims.keys()), "T2": list(t2_optims.keys())})
                    
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
                
                best_t1_name = max(t1_scores, key=t1_scores.get)
                worst_t2_name = min(t2_scores, key=t2_scores.get)
                best_t1_key = best_t1_name.replace("T1_", "")
                worst_t2_key = worst_t2_name.replace("T2_", "")

                # a) Swap
                if t1_scores[best_t1_name] > t2_scores[worst_t2_name] + self.swap_threshold and self.cooldowns[best_t1_key] == 0 and self.active_grace_period is None:
                    print(f"\n  [SWAP] {best_t1_name} ({t1_scores[best_t1_name]:.3f}) beats {worst_t2_name} ({t2_scores[worst_t2_name]:.3f}) by >{self.swap_threshold*100:.0f}%!")
                    reporter.emit("switch", {"action": "Swap", "algo": f"{best_t1_key} <-> {worst_t2_key}", "details": "Demoting T2 algorithm due to low composite score"})
                    
                    import copy
                    self.active_grace_period = {
                        'algo': best_t1_key,
                        'iters_left': 5,
                        'baseline': glob,
                        'snapshot': copy.deepcopy(t2_optims[worst_t2_key]),
                        'target_t1': worst_t2_key
                    }
                    self.cooldowns[worst_t2_key] = 2 * self.k
                    
                    old_t2 = list(t2_optims[worst_t2_key].population)
                    old_t1 = list(active_t1_optims[best_t1_key].population)
                    t2_size_target = t2_optims[worst_t2_key].pop_size
                    
                    new_t2 = _make_optimizer(best_t1_key, self.space, t2_size_target)
                    new_t2.tell(old_t1 + old_t2[len(old_t1):])
                    
                    new_t1 = _make_optimizer(worst_t2_key, self.space, t1_pop_size)
                    new_t1.tell(old_t2[:t1_pop_size])
                    
                    t2_optims[best_t1_key] = new_t2
                    del t2_optims[worst_t2_key]
                    
                    active_t1_optims[worst_t2_key] = new_t1
                    del active_t1_optims[best_t1_key]
                    
                    stagnation_counters[best_t1_key] = 0
                    if worst_t2_key in stagnation_counters: del stagnation_counters[worst_t2_key]
                    
                    reporter.emit("tier_state", {"T1": list(active_t1_optims.keys()), "T2": list(t2_optims.keys())})

                # b) Stagnation
                for k, opt in list(t2_optims.items()):
                    if rw.get(f"T2_{k}", {}).get('trend', 1.0) == 0.0:
                        stagnation_counters[k] = stagnation_counters.get(k, 0) + 1
                        if stagnation_counters[k] >= self.stagnation_limit:
                            t2_optims[k] = self._random_inject(opt, self.inject_pct, dims, f"T2_{k}")
                            stagnation_counters[k] = 0
                    else:
                        stagnation_counters[k] = 0

                # c) Elite Sharing (Tier 2 ONLY)
                print(f"\n  [ELITE SHARE] Pooling top {int(self.elite_pct*100)}% from Tier 2 algorithms and redistributing...")
                reporter.emit("switch", {"action": "Elite Sharing", "algo": "Tier 2 Only", "details": "Swapped top individuals between T2 islands"})
                elite_pool = []
                for k, opt in t2_optims.items():
                    n = max(1, int(opt.pop_size * self.elite_pct))
                    elite_pool.extend(sorted(opt.population, key=lambda x: x.fitness, reverse=True)[:n])
                
                elite_pool.sort(key=lambda x: x.fitness, reverse=True)
                
                for k, opt in t2_optims.items():
                    n = max(1, int(opt.pop_size * self.elite_pct))
                    s = sorted(opt.population, key=lambda x: x.fitness, reverse=True)
                    opt.population = s[:-n] + elite_pool[:n]
                
                print(f"  {_sep('-', 80)}\n")
                
                # Reprint the header cleanly after all checkpoint outputs are finished
                curr_layout = _print_t2_header()

        # ==============================================================
        # FINAL RESULTS
        # ==============================================================
        reporter.emit("step", {"num": 6, "name": "Final Results"})
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
        print(f"\n  Best Hyperparameters:")
        params = _decode_individual(self.best_individual, self.space)
        for k, v in params.items():
            print(f"    * {k:<20}: {v:.4f}" if isinstance(v, float) else f"    * {k:<20}: {v}")
        print(_sep())

        return self.best_individual
