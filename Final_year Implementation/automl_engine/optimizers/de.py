import numpy as np
import random
from .base import BaseOptimizer

class DifferentialEvolution(BaseOptimizer):
    def __init__(self, search_space, pop_size, f=0.7, cr=0.9):
        super().__init__(search_space, pop_size)
        self.f = f
        self.cr = cr
        # The population holds Individual objects
        self.population = []
        # Store parents for the current generation's trial vectors
        self._current_parents = []
        
    def ask(self, num_samples):
        # If population is not yet fully initialized, return random vectors
        if len(self.population) < self.pop_size:
            self._current_parents = []
            return [np.random.rand(self.dimensions) for _ in range(num_samples)]
            
        new_vectors = []
        self._current_parents = []
        
        # In DE, each individual in the population acts as a target vector
        # Since we might be asked for fewer samples than pop_size (e.g. batching),
        # we can just take a random sample of parents to mutate, or iterate.
        # For simplicity, we just use random parents.
        for _ in range(num_samples):
            target_idx = random.randint(0, len(self.population) - 1)
            target_ind = self.population[target_idx]
            
            # Select 3 distinct individuals other than the target
            indices = list(range(len(self.population)))
            indices.remove(target_idx)
            r1, r2, r3 = random.sample(indices, 3)
            
            x1 = self.population[r1].vector
            x2 = self.population[r2].vector
            x3 = self.population[r3].vector
            
            # Mutation
            mutant = x1 + self.f * (x2 - x3)
            
            # Binomial Crossover
            trial = np.copy(target_ind.vector)
            j_rand = random.randint(0, self.dimensions - 1)
            
            for j in range(self.dimensions):
                if random.random() < self.cr or j == j_rand:
                    trial[j] = mutant[j]
                    
            # Ensure boundaries
            trial = np.clip(trial, 0.0, 1.0)
            
            new_vectors.append(trial)
            self._current_parents.append(target_ind)
            
        return new_vectors

    def tell(self, evaluated_individuals):
        # Initial population setup
        if len(self.population) < self.pop_size:
            self.population.extend(evaluated_individuals)
            # Ensure we don't exceed pop_size during initialization
            if len(self.population) > self.pop_size:
                self.population.sort(key=lambda ind: ind.fitness, reverse=True)
                self.population = self.population[:self.pop_size]
            return

        # DE Selection: compare trial vectors against their specific parents
        for i, trial_ind in enumerate(evaluated_individuals):
            if i >= len(self._current_parents):
                break
                
            parent = self._current_parents[i]
            # Replace parent if trial is better or equal
            if trial_ind.fitness >= parent.fitness:
                # We replace the parent's vector and fitness with the trial's
                parent.vector = trial_ind.vector
                parent.fitness = trial_ind.fitness
                parent.source_algorithm = trial_ind.source_algorithm
