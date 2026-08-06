import numpy as np

class BaseOptimizer:
    """
    Abstract base class for all evolutionary algorithms (GA, PSO, DE, CMA-ES).
    All algorithms must operate on scaled [0, 1] float vectors.
    """
    def __init__(self, search_space, pop_size):
        self.search_space = search_space
        self.pop_size = pop_size
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
