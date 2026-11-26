import numpy as np
from .base import BaseOptimizer
from evosynth.config import get_settings


class DifferentialEvolution(BaseOptimizer):
    """
    DE/rand/1/bin with one-to-one (greedy) selection (Phase 2.2): each trial vector competes only
    with the target it was built from and replaces it if it is at least as good. Targets are
    visited round-robin, so asking for fewer samples than pop_size still covers the whole
    population over successive asks.
    """

    def __init__(self, search_space, pop_size, rng=None):
        super().__init__(search_space, pop_size, rng)
        d = get_settings().optimizers.de  # [optimizers.de] in settings/optimizers.toml
        self.F = d.F    # Mutation factor
        self.CR = d.CR  # Crossover probability
        self._cursor = 0        # next target index (round-robin)
        self._targets = []      # target index of each trial returned by the last ask

    def ask(self, num_samples):
        self._targets = []
        # DE/rand/1 needs the target plus 3 distinct others; until then, sample randomly
        if len(self.population) < 4:
            return [self.rng.random(self.dimensions) for _ in range(num_samples)]

        pop_vecs = [np.copy(ind.vector) for ind in self.population]
        pop_len = len(pop_vecs)
        new_vectors = []
        for _ in range(num_samples):
            target_idx = self._cursor % pop_len
            self._cursor += 1
            x_i = pop_vecs[target_idx]

            # 3 distinct random individuals, all different from the target
            candidates = [idx for idx in range(pop_len) if idx != target_idx]
            a_idx, b_idx, c_idx = self.rng.choice(candidates, 3, replace=False)
            a, b, c = pop_vecs[a_idx], pop_vecs[b_idx], pop_vecs[c_idx]

            # Mutation
            v = np.clip(a + self.F * (b - c), 0.0, 1.0)

            # Binomial crossover
            u = np.copy(x_i)
            j_rand = self.rng.integers(self.dimensions)
            for j in range(self.dimensions):
                if self.rng.random() < self.CR or j == j_rand:
                    u[j] = v[j]

            self._targets.append(target_idx)
            new_vectors.append(u)
        return new_vectors

    def tell(self, evaluated_individuals):
        evaluated_individuals = list(evaluated_individuals)
        if not self._targets:
            # Seed population (or random samples while the population is too small): keep the best
            self.population.extend(evaluated_individuals)
            self.population.sort(key=lambda ind: ind.fitness, reverse=True)
            self.population = self.population[:self.pop_size]
            return
        # One-to-one selection: trial j competes only with its own target
        for target_idx, trial in zip(self._targets, evaluated_individuals):
            if trial.fitness >= self.population[target_idx].fitness:
                self.population[target_idx] = trial
        self._targets = []

    def inject(self, individuals):
        self._targets = []
        super().inject(individuals)
