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

class XGBoostEvaluator:
    """
    Evaluates hyperparameter configurations by training an XGBoost model
    on the Breast Cancer classification dataset.
    """
    def __init__(self):
        from .search_space import get_xgboost_search_space
        from sklearn.datasets import load_breast_cancer
        
        self.space = get_xgboost_search_space()
        
        # Load Breast Cancer dataset once
        data = load_breast_cancer()
        self.X = data.data
        self.y = data.target
        
    def evaluate(self, individual):
        """
        Receives an Individual (with a [0, 1] scaled vector),
        decodes it into real hyperparameter values, trains XGBoost, and returns accuracy.
        """
        import xgboost as xgb
        from sklearn.model_selection import cross_val_score
        
        valid_vec = np.copy(individual.vector)
        for i, hp_name in enumerate(list(self.space.keys())):
            hp = self.space.get_hyperparameter(hp_name)
            if isinstance(hp, CSH.CategoricalHyperparameter):
                num_choices = len(hp.choices)
                idx = int(np.floor(valid_vec[i] * num_choices))
                idx = min(idx, num_choices - 1)
                valid_vec[i] = float(idx)
                
        config = Configuration(self.space, vector=valid_vec)
        
        # Extract parameters
        lr = config["learning_rate"]
        n_est = config["n_estimators"]
        depth = config["max_depth"]
        sub = config["subsample"]
        booster = config["booster"]
        
        # Initialize model
        model = xgb.XGBClassifier(
            learning_rate=lr,
            n_estimators=n_est,
            max_depth=depth,
            subsample=sub,
            booster=booster,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1 # Limit threads so the swarms can run fast
        )
        
        # Perform 3-fold cross validation
        scores = cross_val_score(model, self.X, self.y, cv=3, scoring="accuracy")
        
        # Return accuracy as percentage (0-100)
        mean_accuracy = scores.mean() * 100.0
        return mean_accuracy
