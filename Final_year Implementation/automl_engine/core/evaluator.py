import numpy as np
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import Configuration
from .search_space import get_dummy_search_space

class DummyObjectiveFunction:
    """
    A mathematical dummy function that simulates an ML model.
    Evaluates in microseconds and has a known global optimum.
    """
    def __init__(self):
        self.space = get_dummy_search_space()
        
        # The "Secret" Perfect Answer we want the optimizers to find
        self.perfect_lr = 0.05
        self.perfect_layers = 4
        self.perfect_opt = "Adam"
        self.perfect_dropout = 0.2
        
    def evaluate(self, individual):
        """
        Receives an Individual (with a [0, 1] scaled vector),
        decodes it into real hyperparameter values, and scores it.
        """
        # Map our [0, 1] continuous vector to a valid ConfigSpace vector
        valid_vec = np.copy(individual.vector)
        for i, hp_name in enumerate(list(self.space.keys())):
            hp = self.space.get_hyperparameter(hp_name)
            if isinstance(hp, CSH.CategoricalHyperparameter):
                # Categoricals expect integer indices (as floats) in newer ConfigSpace
                num_choices = len(hp.choices)
                idx = int(np.floor(valid_vec[i] * num_choices))
                idx = min(idx, num_choices - 1) # Handle exactly 1.0
                valid_vec[i] = float(idx)
                
        # Decode the valid vector back into real hyperparameters
        config = Configuration(self.space, vector=valid_vec)
        
        lr = config["learning_rate"]
        layers = config["num_layers"]
        opt = config["optimizer"]
        dropout = config["dropout"]
        
        # 2. Calculate "Error" (Distance from perfect answer)
        # Using simple mean absolute error for numerical values
        error_lr = abs(lr - self.perfect_lr) * 500  # Scale up small float
        error_layers = abs(layers - self.perfect_layers) * 5
        error_dropout = abs(dropout - self.perfect_dropout) * 100
        
        # Categorical penalty
        error_opt = 0 if opt == self.perfect_opt else 20
        
        total_error = error_lr + error_layers + error_dropout + error_opt
        
        # 3. Return Mock Accuracy (100 - error)
        # Max score is 100 if the vector perfectly hits the secret answer
        mock_accuracy = 100.0 - total_error
        
        return max(0.0, mock_accuracy)
