import numpy as np
from automl_engine.engine.population import PopulationManager
from automl_engine.engine.tier_manager import TierManager
from automl_engine.engine.composite import CompositeScoringEngine
from automl_engine.engine.ucb1 import UCB1Bandit
from automl_engine.optimizers.ga import GeneticAlgorithm
from automl_engine.optimizers.pso import ParticleSwarm
from automl_engine.core.individual import Individual

class DynamicOptimizer:
    def __init__(self, search_space, evaluator, budget=100, k_iters=5):
        self.space = search_space
        self.eval = evaluator
        self.budget = budget
        self.k = k_iters
        
        self.composite = CompositeScoringEngine()
        self.bandit = UCB1Bandit()
        
        # Available algorithms pool
        self.strategies = {
            'GA': GeneticAlgorithm,
            'PSO': ParticleSwarm
        }
        
        self.evals_used = 0
        self.best_individual = None
        
    def _evaluate_batch(self, vectors, algo_name):
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
            
            # Global best tracking
            if self.best_individual is None or fitness > self.best_individual.fitness:
                self.best_individual = ind
                print(f"[*] New Best! [{algo_name}] Score: {fitness:.2f}/100")
                
        return inds

    def run(self):
        print(f"--- Starting Dynamic AutoML Engine (Budget: {self.budget} evals) ---")
        
        pop_manager = PopulationManager(self.space, total_size=50)
        
        print("[Phase 1] Generating and scoring initial population of 50...")
        init_pop = []
        for _ in range(50):
            if self.evals_used >= self.budget: break
            vec = np.random.rand(len(self.space.get_hyperparameters()))
            inds = self._evaluate_batch([vec], 'RandomInit')
            init_pop.extend(inds)
            
        tier_manager = TierManager(init_pop)
        
        print("[Phase 2] Assigning Tier 1 (Competition) and Tier 2 (Working Islands)...")
        # Setup T1 Optimizers (GA and PSO)
        t1_size = tier_manager.t1_size // 2
        t1_optim = {
            'GA': self.strategies['GA'](self.space, t1_size),
            'PSO': self.strategies['PSO'](self.space, t1_size)
        }
        t1_optim['GA'].tell(tier_manager.allocate_t1_batch(t1_size))
        t1_optim['PSO'].tell(tier_manager.allocate_t1_batch(t1_size))
        
        # Setup T2 Optimizers (For this initial run, we assign GA to the large working island)
        t2_size = tier_manager.t2_size
        t2_optim = {
            'GA': self.strategies['GA'](self.space, t2_size)
        }
        t2_optim['GA'].tell(tier_manager.allocate_t2_batch(t2_size))

        iteration = 0
        while self.evals_used < self.budget:
            iteration += 1
            
            # 1. Evolve Tier 1 (Small Batches)
            for name, opt in t1_optim.items():
                vecs = opt.ask(t1_size)
                inds = self._evaluate_batch(vecs, f"T1_{name}")
                if inds: opt.tell(inds)
                
            # 2. Evolve Tier 2 (Large Working Batch)
            for name, opt in t2_optim.items():
                vecs = opt.ask(t2_size)
                inds = self._evaluate_batch(vecs, f"T2_{name}")
                if inds: opt.tell(inds)
                
            if self.evals_used >= self.budget:
                break
                
            # 3. Engine Mechanisms (Triggered every k iterations)
            if iteration % self.k == 0:
                print(f"--- Iteration {iteration} Reached (Evaluations: {self.evals_used}/{self.budget}) ---")
                
                # Mock a Composite Scoring Event
                self.bandit.update('GA', np.random.random())
                self.bandit.update('PSO', np.random.random())
                
                # Let UCB1 decide what to do next based on the fake composite scores
                next_algo = self.bandit.select_next(['GA', 'PSO'])
                print(f"[Engine] UCB1 Bandit suggests giving resources to: {next_algo}")
                
        print("--- Optimization Completed ---")
        return self.best_individual
