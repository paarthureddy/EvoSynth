import numpy as np
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import Configuration
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
import warnings

class PathologicalXGBoostEvaluator:
    """
    Evaluates hyperparameter configurations on a highly noisy, 
    pathological synthetic dataset designed to force optimization 
    algorithms to stagnate and trigger advanced switching safeguards.
    """
    def __init__(self):
        from .search_space import get_xgboost_search_space
        warnings.filterwarnings("ignore", category=UserWarning)

        self.space = get_xgboost_search_space()

        # Create a highly noisy dataset
        # flip_y=0.3 adds 30% random label noise, making it incredibly 
        # easy for models to overfit and algorithms to get trapped in local minima.
        self.X, self.y = make_classification(
            n_samples=2000,
            n_features=20,
            n_informative=5,
            n_redundant=10,
            n_classes=2,
            flip_y=0.3,
            random_state=42
        )

    def evaluate(self, individual):
        valid_vec = np.copy(individual.vector)
        for i, hp_name in enumerate(list(self.space.keys())):
            hp = self.space.get_hyperparameter(hp_name)
            if isinstance(hp, CSH.CategoricalHyperparameter):
                num_choices = len(hp.choices)
                idx = int(np.floor(valid_vec[i] * num_choices))
                idx = min(idx, num_choices - 1)
                valid_vec[i] = float(idx)

        config = Configuration(self.space, vector=valid_vec)

        lr = config["learning_rate"]
        n_est = config["n_estimators"]
        depth = config["max_depth"]
        sub = config["subsample"]
        booster = config["booster"]
        colsample = config["colsample_bytree"]
        reg_alpha = config["reg_alpha"]

        model = xgb.XGBClassifier(
            learning_rate=lr,
            n_estimators=n_est,
            max_depth=depth,
            subsample=sub,
            booster=booster,
            colsample_bytree=colsample,
            reg_alpha=reg_alpha,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1
        )

        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, self.X, self.y, cv=3, scoring="accuracy")
        return scores.mean() * 100.0
