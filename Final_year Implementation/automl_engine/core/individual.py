class Individual:
    """
    Represents a single hyperparameter configuration in the population.
    The vector is a scaled numpy array [0, 1] allowing evolutionary math.
    """
    def __init__(self, vector):
        # The latent continuous vector representation
        self.vector = vector 
        
        # The raw fitness score from the objective function
        self.fitness = None
        
        # Track which algorithm spawned this individual
        self.source_algorithm = None

    def set_fitness(self, fitness):
        self.fitness = fitness
