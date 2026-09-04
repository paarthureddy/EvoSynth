import numpy as np
import cma
from .base import BaseOptimizer


class CMAESOptimizer(BaseOptimizer):
    def __init__(self, search_space, pop_size):
        super().__init__(search_space, pop_size)
        self.es = None
        self.pending_solutions = []
        self._eval_buffer_x = []
        self._eval_buffer_y = []

    def ask(self, num_samples):
        if self.es is None:
            # Initialize from population mean if possible (e.g. from Tier split or elite sharing)
            if len(self.population) > 0:
                mean = np.mean([ind.vector for ind in self.population], axis=0)
            else:
                mean = 0.5 * np.ones(self.dimensions)

            # Initialize CMA-ES (bounded to [0, 1])
            self.es = cma.CMAEvolutionStrategy(
                mean,
                0.25,  # Initial step size
                {'bounds': [0, 1], 'popsize': max(
                    4, self.pop_size), 'verbose': -9}
            )

        # Draw samples from the CMA-ES distribution
        new_vectors = []
        while len(new_vectors) < num_samples:
            if not self.pending_solutions:
                # CMA-ES asks for a full generation at once
                self.pending_solutions = self.es.ask()
            new_vectors.append(self.pending_solutions.pop(0))

        # Ensure strict bounds
        return [np.clip(v, 0.0, 1.0) for v in new_vectors]

    def tell(self, evaluated_individuals):
        # CMA-ES expects a full generation to be told at once, and it minimizes by default
        for ind in evaluated_individuals:
            self._eval_buffer_x.append(np.copy(ind.vector))
            # Invert fitness to minimize
            self._eval_buffer_y.append(-ind.fitness)

        # If we have a full generation and ES is initialized, tell the ES object
        if self.es is not None:
            while len(self._eval_buffer_x) >= self.es.popsize:
                batch_x = self._eval_buffer_x[:self.es.popsize]
                batch_y = self._eval_buffer_y[:self.es.popsize]

                try:
                    self.es.tell(batch_x, batch_y)
                except Exception:
                    # In rare cases, matrix math can crash due to singularity in tight bounds
                    pass

                self._eval_buffer_x = self._eval_buffer_x[self.es.popsize:]
                self._eval_buffer_y = self._eval_buffer_y[self.es.popsize:]

        # Keep our internal population updated for Tier tracking and Elite Sharing
        self.population.extend(evaluated_individuals)
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        self.population = self.population[:self.pop_size]
