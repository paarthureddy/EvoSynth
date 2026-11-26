import numpy as np
from .base import BaseOptimizer
from evosynth.config import get_settings


class GeneticAlgorithm(BaseOptimizer):
    def __init__(self, search_space, pop_size, mutation_rate=None, rng=None):
        super().__init__(search_space, pop_size, rng)
        g = get_settings().optimizers.ga  # [optimizers.ga] in settings/optimizers.toml
        self.mutation_rate = g.mutation_rate if mutation_rate is None else mutation_rate
        self.mutation_sigma = g.mutation_sigma
        self.tournament_size = g.tournament_size

    def ask(self, num_samples):
        # If population is empty or too small, return random vectors
        if len(self.population) < 2:
            return [self.rng.random(self.dimensions) for _ in range(num_samples)]

        new_vectors = []
        for _ in range(num_samples):
            # Tournament selection (size = tournament_size)
            k = min(self.tournament_size, len(self.population))
            candidates1 = [self.population[i] for i in self.rng.choice(len(self.population), k, replace=False)]
            candidates2 = [self.population[i] for i in self.rng.choice(len(self.population), k, replace=False)]

            p1 = max(candidates1, key=lambda ind: ind.fitness).vector
            p2 = max(candidates2, key=lambda ind: ind.fitness).vector

            # Continuous Arithmetic Crossover
            alpha = self.rng.random(self.dimensions)
            child = alpha * p1 + (1 - alpha) * p2

            # Mutation (Gaussian noise)
            for i in range(self.dimensions):
                if self.rng.random() < self.mutation_rate:
                    child[i] += self.rng.normal(0, self.mutation_sigma)

            # Ensure boundaries
            child = np.clip(child, 0.0, 1.0)
            new_vectors.append(child)

        return new_vectors

    def tell(self, evaluated_individuals):
        self.population.extend(evaluated_individuals)
        # Keep only the fittest individuals up to pop_size
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        self.population = self.population[:self.pop_size]
