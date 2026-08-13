import numpy as np
import time
from automl_engine.engine.tier_manager import TierManager
from automl_engine.engine.composite import CompositeScoringEngine
from automl_engine.engine.ucb1 import UCB1Bandit
from automl_engine.optimizers.ga import GeneticAlgorithm
from automl_engine.optimizers.pso import ParticleSwarm
from automl_engine.optimizers.de import DifferentialEvolution
from automl_engine.optimizers.cmaes import CMAES
from automl_engine.core.individual import Individual


def _sep(char="=", width=70):
    return char * width

def _header(title, char="=", width=70):
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
    """Factory: returns a fresh optimizer instance."""
    if algo_key == 'GA': return GeneticAlgorithm(space, pop_size)
    if algo_key == 'PSO': return ParticleSwarm(space, pop_size)
    if algo_key == 'DE': return DifferentialEvolution(space, pop_size)
    if algo_key == 'CMAES': return CMAES(space, pop_size)
    return GeneticAlgorithm(space, pop_size)


class DynamicOptimizer:
    """
    Full hybrid controller implementing the 4-algorithm strategy switching.
    """

    def __init__(self, search_space, evaluator,
                 budget=300, t1_iters=20, k_iters=5,
                 swap_threshold=0.05, stagnation_limit=2,
                 elite_pct=0.20, inject_pct=0.20):
        self.space = search_space
        self.eval = evaluator
        self.budget = budget
        self.t1_iters = t1_iters # Currently unused, as warm-up is 5 iters as per PDF
        self.k = k_iters
        self.swap_threshold = swap_threshold
        self.stagnation_limit = stagnation_limit
        self.elite_pct = elite_pct
        self.inject_pct = inject_pct

        self.composite_engine = CompositeScoringEngine()
        self.bandit = UCB1Bandit()
        
        self.all_algos = ['GA', 'PSO', 'DE', 'CMAES']

        self.evals_used = 0
        self.best_individual = None
        self._algo_stats = {alg: {'best': 0.0, 'prev_best': 0.0} for alg in self.all_algos}

    def _evaluate_batch(self, vectors, algo_name):
        algo_key = algo_name.split("_")[-1]
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

    def _compute_composite(self, algo_pops):
        algo_data = {}
        for name, pop in algo_pops.items():
            if not pop: continue
            fitnesses = [ind.fitness for ind in pop if ind.fitness is not None]
            if not fitnesses: continue
            best = max(fitnesses)
            spread = max(fitnesses) - min(fitnesses) + 1e-9
            diversity = min(float(np.std(fitnesses)) / spread, 1.0) if len(fitnesses) > 1 else 0.0
            
            algo_key = name.split("_")[-1]
            prev = self._algo_stats.get(algo_key, {}).get('prev_best', 0.0)
            trend = max(best - prev, 0.0)
            self._algo_stats[algo_key]['prev_best'] = best
            
            algo_data[name] = {
                'fitness': best, 'speed': trend,
                'efficiency': trend / float(self.k), 'diversity': diversity, 'trend': trend
            }
        return self.composite_engine.calculate_scores(algo_data), algo_data

    def _print_scoring_table(self, algo_pops, scores, raw):
        print(f"\n  {'Algorithm':<12} {'Best Acc':>9} {'Diversity':>10} {'Composite':>10}")
        print(f"  {'-'*12} {'-'*9} {'-'*10} {'-'*10}")
        # Sort by composite descending
        sorted_names = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        for name in sorted_names:
            if name not in raw: continue
            print(f"  {name:<12} {raw[name]['fitness']:>8.2f}%"
                  f" {raw[name]['diversity']:>10.3f}"
                  f" {scores.get(name, 0.0):>10.3f}")

    def _random_inject(self, optimizer, inject_pct, dims, tag):
        n_inject = max(1, int(len(optimizer.population) * inject_pct))
        print(f"  [INJECT] Replacing bottom {n_inject} individuals in {tag} with fresh random vectors.")
        fresh_vecs = [np.random.rand(dims) for _ in range(n_inject)]
        fresh_inds = self._evaluate_batch(fresh_vecs, tag)
        optimizer.population.sort(key=lambda x: x.fitness, reverse=True)
        optimizer.population = optimizer.population[:-n_inject] + fresh_inds
        return optimizer

    def run(self):
        total_start = time.time()
        dims = len(list(self.space.keys()))

        print(_sep())
        print(_header("STEP 1: SEARCH SPACE DEFINITION"))
        print(_sep())
        print(f"  Hyperparameter Space : {dims} dimensions")
        print(f"  Budget             : {self.budget} evals")
        print(f"  Scoring Interval k : {self.k} iters")

        print(f"\n{_sep()}")
        print(_header("STEP 2: INITIALIZATION (50 Random Individuals)"))
        print(_sep())
        print("  Generating 50 random configs and evaluating...\n")
        init_pop = []
        for _ in range(50):
            if self.evals_used >= self.budget: break
            init_pop.extend(self._evaluate_batch([np.random.rand(dims)], 'RandomInit'))
        
        tier_manager = TierManager(init_pop, t1_ratio=0.4)
        print(f"  [OK] {len(init_pop)} individuals evaluated")
        print(f"  T1 (Competition)  : {tier_manager.t1_size} individuals")
        print(f"  T2 (Working)      : {tier_manager.t2_size} individuals")

        print(f"\n{_sep()}")
        print(_header("STEP 3: TIER 1 WARM-UP (5 Iterations)"))
        print(_sep())
        
        # T1 receives 20 individuals, split among 4 strategies (5 each)
        t1_batch_size = tier_manager.t1_size // 4
        t1_optims = {}
        for alg in self.all_algos:
            t1_optims[alg] = _make_optimizer(alg, self.space, t1_batch_size)
            t1_optims[alg].tell(tier_manager.allocate_t1_batch(t1_batch_size))
            
        for t1_iter in range(1, 6):
            if self.evals_used >= self.budget: break
            for alg, opt in t1_optims.items():
                inds = self._evaluate_batch(opt.ask(t1_batch_size), f"T1_{alg}")
                if inds: opt.tell(inds)
        print("  [OK] Warm-up completed for GA, PSO, DE, CMAES.")

        print(f"\n{_sep()}")
        print(_header("STEP 4: INITIAL PROMOTION TO TIER 2"))
        print(_sep())
        
        t1_pops = {f"T1_{k}": list(v.population) for k, v in t1_optims.items()}
        scores, raw = self._compute_composite(t1_pops)
        self._print_scoring_table(t1_pops, scores, raw)
        
        sorted_algos = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        t2_names = sorted_algos[:2]
        t1_names = sorted_algos[2:]
        
        print("\n  [PROMOTION RESULTS]")
        for name in t2_names:
            print(f"  PROMOTE : [{name}] --> Tier 2")
        for name in t1_names:
            print(f"  RETAIN  : [{name}] stays in Tier 1")
            
        # Distribute 30 T2 individuals to the two promoted strategies (15 each)
        t2_batch_size = tier_manager.t2_size // 2
        t2_optims = {}
        for name in t2_names:
            alg_key = name.split("_")[-1]
            t2_optims[alg_key] = _make_optimizer(alg_key, self.space, t2_batch_size)
            # Seed with their existing T1 individuals + the unassigned T2 pool
            t2_optims[alg_key].tell(list(t1_optims[alg_key].population) + tier_manager.allocate_t2_batch(t2_batch_size - t1_batch_size))
            del t1_optims[alg_key]
            
        t2_keys = list(t2_optims.keys())
        t1_keys = list(t1_optims.keys())
        
        t2_stagnation = {k: 0 for k in t2_keys}

        print(f"\n{_sep()}")
        print(_header(f"STEP 5: MAIN LOOP [T2: {t2_keys} | T1: {t1_keys}]"))
        print(_sep())
        print(f"  {'Iter':>4}  {'Status':<18}  {'T2-1 Best':>9}  {'T2-2 Best':>9}  {'Global':>9}  {'Evals':>9}")
        print(f"  {'-'*4}  {'-'*18}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")

        iteration = 5
        while self.evals_used < self.budget:
            iteration += 1
            prev_best = self.best_individual.fitness if self.best_individual else 0.0

            # Evolve T2
            for alg, opt in t2_optims.items():
                inds = self._evaluate_batch(opt.ask(t2_batch_size), f"T2_{alg}")
                if inds: opt.tell(inds)

            # Evolve T1
            for alg, opt in t1_optims.items():
                inds = self._evaluate_batch(opt.ask(t1_batch_size), f"T1_{alg}")
                if inds: opt.tell(inds)

            if self.evals_used >= self.budget: break

            glob = self.best_individual.fitness if self.best_individual else 0.0
            
            t2_b1 = max((i.fitness for i in t2_optims[t2_keys[0]].population), default=0.0)
            t2_b2 = max((i.fitness for i in t2_optims[t2_keys[1]].population), default=0.0) if len(t2_keys)>1 else 0.0
            
            new_b = glob > prev_best
            status = "*** NEW BEST ***  " if new_b else "                  "

            print(f"  {iteration:>4}  {status}  {t2_b1:>8.2f}%  {t2_b2:>8.2f}%  {glob:>8.2f}%  {self.evals_used:>4}/{self.budget}")
            if new_b:
                params = _decode_individual(self.best_individual, self.space)
                print("         `-- " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()))

            if iteration % self.k == 0:
                print(f"\n  {_sep('-', 60)}")
                print(f"  {_header('SWITCHING STRATEGY CHECKPOINT', '-', 60)}")
                
                # a) Elite Sharing between T2 islands
                if len(t2_keys) >= 2:
                    print(f"  [ELITE SHARE] Swapping top {int(self.elite_pct*100)}% between T2_{t2_keys[0]} and T2_{t2_keys[1]}")
                    pop_a = t2_optims[t2_keys[0]].population
                    pop_b = t2_optims[t2_keys[1]].population
                    new_a, new_b = TierManager.share_elites(pop_a, pop_b, top_pct=self.elite_pct)
                    t2_optims[t2_keys[0]].population = new_a
                    t2_optims[t2_keys[1]].population = new_b
                
                # Scoring
                all_pops = {f"T2_{k}": list(v.population) for k, v in t2_optims.items()}
                all_pops.update({f"T1_{k}": list(v.population) for k, v in t1_optims.items()})
                
                sc, rw = self._compute_composite(all_pops)
                self._print_scoring_table(all_pops, sc, rw)
                
                for k in self.all_algos:
                    self.bandit.register_algorithm(k)
                    val = sc.get(f"T2_{k}", sc.get(f"T1_{k}", 0.0))
                    self.bandit.update(k, val)

                # b) Challenge Check (Demotion/Promotion Swap)
                t2_scores = {k: sc.get(f"T2_{k}", 0.0) for k in t2_keys}
                t1_scores = {k: sc.get(f"T1_{k}", 0.0) for k in t1_keys}
                
                min_t2_key = min(t2_scores, key=t2_scores.get)
                max_t1_key = max(t1_scores, key=t1_scores.get)
                
                action_taken = False
                
                if t1_scores[max_t1_key] > t2_scores[min_t2_key] + self.swap_threshold:
                    print(f"\n  [SWAP] T1-{max_t1_key} composite ({t1_scores[max_t1_key]:.3f}) beats T2-{min_t2_key} ({t2_scores[min_t2_key]:.3f})!")
                    print(f"  [SWAP] Demoting {min_t2_key} to T1. Promoting {max_t1_key} to T2.")
                    
                    old_t2_pop = list(t2_optims[min_t2_key].population)
                    old_t1_pop = list(t1_optims[max_t1_key].population)
                    
                    new_t2_optim = _make_optimizer(max_t1_key, self.space, t2_batch_size)
                    new_t2_optim.tell(old_t1_pop + [ind for ind in old_t2_pop if ind not in old_t1_pop][:t2_batch_size - len(old_t1_pop)]) # Ensure proper sizing
                    new_t1_optim = _make_optimizer(min_t2_key, self.space, t1_batch_size)
                    new_t1_optim.tell(old_t2_pop[:t1_batch_size])
                    
                    t2_optims[max_t1_key] = new_t2_optim
                    del t2_optims[min_t2_key]
                    
                    t1_optims[min_t2_key] = new_t1_optim
                    del t1_optims[max_t1_key]
                    
                    t2_keys = list(t2_optims.keys())
                    t1_keys = list(t1_optims.keys())
                    t2_stagnation[max_t1_key] = 0
                    if min_t2_key in t2_stagnation: del t2_stagnation[min_t2_key]
                    action_taken = True
                    
                # c) Stagnation Detection + Injection / UCB1
                if not action_taken:
                    for t2_k in t2_keys:
                        if rw.get(f"T2_{t2_k}", {}).get('trend', 1.0) == 0.0:
                            t2_stagnation[t2_k] = t2_stagnation.get(t2_k, 0) + 1
                            print(f"\n  [STAGNATION] T2-{t2_k} trend = 0 for {t2_stagnation[t2_k]}/{self.stagnation_limit} checks.")
                            if t2_stagnation[t2_k] >= self.stagnation_limit:
                                t2_optims[t2_k] = self._random_inject(t2_optims[t2_k], self.inject_pct, dims, f"T2_{t2_k}")
                                
                                # If stagnation persists > stagnation_limit + 1, UCB1 force swap
                                if t2_stagnation[t2_k] > self.stagnation_limit:
                                    ucb_choice = self.bandit.select_next(t1_keys)
                                    print(f"\n  [UCB1 FORCE] Exploration bonus favors [{ucb_choice}]! Force-promoting it to replace {t2_k}.")
                                    # Swap logic
                                    new_t2 = _make_optimizer(ucb_choice, self.space, t2_batch_size)
                                    new_t2.tell(list(t1_optims[ucb_choice].population))
                                    new_t1 = _make_optimizer(t2_k, self.space, t1_batch_size)
                                    new_t1.tell(list(t2_optims[t2_k].population)[:t1_batch_size])
                                    
                                    t2_optims[ucb_choice] = new_t2
                                    del t2_optims[t2_k]
                                    t1_optims[t2_k] = new_t1
                                    del t1_optims[ucb_choice]
                                    
                                    t2_keys = list(t2_optims.keys())
                                    t1_keys = list(t1_optims.keys())
                                    t2_stagnation[ucb_choice] = 0
                                    del t2_stagnation[t2_k]
                                    break
                        else:
                            t2_stagnation[t2_k] = 0
                            
                print(f"  {_sep('-', 60)}\n")
                print(f"  {'Iter':>4}  {'Status':<18}  {'T2-1 Best':>9}  {'T2-2 Best':>9}  {'Global':>9}  {'Evals':>9}")
                print(f"  {'-'*4}  {'-'*18}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")

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
            print(f"    * {k:<20}: {v:.4f}" if isinstance(v, float) else f"    * {k:<20}: {v}")
        print(_sep())

        return self.best_individual
