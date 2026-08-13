import numpy as np
from .base import BaseOptimizer

class DifferentialEvolution(BaseOptimizer):
    def __init__(self, search_space, pop_size):
        super().__init__(search_space, pop_size)
        self.F = 0.8  # Mutation factor
        self.CR = 0.7 # Crossover probability
        self.population_vectors = []
        self.best_score = -float('inf')

    def ask(self, num_samples):
        # If no population yet (from tell()), just generate randomly
        if len(self.population) < 3:
            return [np.random.rand(self.dimensions) for _ in range(num_samples)]
            
        new_vectors = []
        pop_vecs = [np.copy(ind.vector) for ind in self.population]
        pop_len = len(pop_vecs)
        
        for i in range(num_samples):
            # DE/rand/1/bin strategy
            target_idx = i % pop_len
            x_i = pop_vecs[target_idx]
            
            # Select 3 distinct random individuals different from i
            candidates = [idx for idx in range(pop_len) if idx != target_idx]
            if len(candidates) >= 3:
                a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            else:
                a_idx, b_idx, c_idx = np.random.choice(range(pop_len), 3, replace=True)
                
            a, b, c = pop_vecs[a_idx], pop_vecs[b_idx], pop_vecs[c_idx]
            
            # Mutation
            v = a + self.F * (b - c)
            v = np.clip(v, 0.0, 1.0)
            
            # Crossover (Binomial)
            u = np.copy(x_i)
            j_rand = np.random.randint(self.dimensions)
            for j in range(self.dimensions):
                if np.random.rand() < self.CR or j == j_rand:
                    u[j] = v[j]
                    
            new_vectors.append(u)
            
        return new_vectors

    def tell(self, evaluated_individuals):
        # Extend and sort the population
        self.population.extend(evaluated_individuals)
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        # Keep only the top pop_size individuals (this acts as the DE selection step for a continuous population)
        self.population = self.population[:self.pop_size]
