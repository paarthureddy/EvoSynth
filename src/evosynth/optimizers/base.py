import numpy as np

from evosynth.seeding import stream


class BaseOptimizer:
    """
    Abstract base class for all evolutionary algorithms (GA, PSO, DE, CMA-ES).
    All algorithms must operate on scaled [0, 1] float vectors.
    """

    def __init__(self, search_space, pop_size, rng=None):
        self.search_space = search_space
        self.pop_size = pop_size
        # Own random generator (Phase 0). The controller passes one per optimizer instance;
        # used standalone, the optimizer derives one from run.seed and its class name.
        self.rng = rng if rng is not None else stream("optimizer", type(self).__name__)
        self.dimensions = len(search_space.get_hyperparameters())
        self.population = []

    def ask(self, num_samples):
        """
        Generates 'num_samples' new hyperparameter vectors (numpy arrays [0, 1]).
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def tell(self, evaluated_individuals):
        """
        Receives a list of Individual objects with fitness scores attached.
        Updates internal state (e.g., Gbest for PSO, Elite pool for GA).
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def inject(self, individuals):
        """
        Replace the worst members of the population with already-evaluated `individuals`
        (random injection on stagnation, elite sharing between Tier 2 islands). Phase 1.8.
        Subclasses with internal search state (PSO particles, CMA-ES distribution) also update it.
        """
        individuals = list(individuals)
        if not individuals:
            return
        ranked = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
        keep = ranked[:max(0, len(ranked) - len(individuals))]
        self.population = sorted(keep + individuals, key=lambda ind: ind.fitness, reverse=True)[:self.pop_size]
