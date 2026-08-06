import numpy as np
import random
from automl_engine.core.individual import Individual

class PopulationManager:
    """
    Handles the generation and initial distribution of the population.
    """
    def __init__(self, search_space, total_size=50):
        self.search_space = search_space
        self.total_size = total_size
        self.dimensions = len(search_space.get_hyperparameters())
        self.individuals = []
        
    def generate_initial_population(self, evaluator):
        """
        Generates random individuals, evaluates their initial fitness, 
        and returns the scored population.
        """
        self.individuals = []
        for _ in range(self.total_size):
            # Generate random scaled vector [0, 1]
            vec = np.random.rand(self.dimensions)
            ind = Individual(vec)
            
            # Evaluate immediately for initial baseline
            fitness = evaluator.evaluate(ind)
            ind.set_fitness(fitness)
            
            self.individuals.append(ind)
            
        return self.individuals
