import numpy as np
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import Configuration
from .search_space import get_cnn_search_space, get_xgboost_search_space
import warnings


class CNNDummyEvaluator:
    """
    Fast dummy evaluator that simulates CNN accuracy on CIFAR-10.
    Evaluates in microseconds. The 'perfect' config is the hidden optimum
    the algorithms must discover through evolution.
    """

    PERFECT = {
        "num_layers": 8,
        "filters": 192,
        "kernel": 3,         # 3x3 wins
        "learning_rate": 0.00015,
        "optimizer": "SGD",  # matches PDF final output
    }

    def __init__(self):
        self.space = get_cnn_search_space()
        self._rng = np.random.default_rng(0)  # reproducible noise

    def evaluate(self, individual):
        vec = np.copy(individual.vector)
        space = self.space

        # Map [0,1] vector → valid ConfigSpace vector (categoricals need int index)
        for i, hp_name in enumerate(list(space.keys())):
            hp = space.get_hyperparameter(hp_name)
            if isinstance(hp, CSH.CategoricalHyperparameter):
                n = len(hp.choices)
                idx = min(int(np.floor(vec[i] * n)), n - 1)
                vec[i] = float(idx)

        config = Configuration(space, vector=vec)

        l       = config["num_layers"]      # 3..10
        f       = config["filters"]         # 32..256
        k       = config["kernel"]          # 3 or 5
        lr      = config["learning_rate"]   # 0.0001..0.01
        opt     = config["optimizer"]       # Adam / SGD / RMSprop

        p = self.PERFECT

        # Penalise distance from perfect config
        err_l   = abs(l  - p["num_layers"]) * 1.5
        err_f   = abs(f  - p["filters"])    / 64.0 * 3.0
        err_k   = 0.0 if k == p["kernel"] else 2.0
        err_lr  = abs(np.log(lr) - np.log(p["learning_rate"])) * 5.0
        err_opt = 0.0 if opt == p["optimizer"] else 4.0

        total_err = err_l + err_f + err_k + err_lr + err_opt

        # Base accuracy starts at ~70% for a random config, rises toward ~91%
        base_acc = 70.0 + (30.0 / (1.0 + total_err))

        # Add small evaluation noise (±0.5%) to simulate real training variance
        noise = self._rng.uniform(-0.5, 0.5)
        return round(float(np.clip(base_acc + noise, 0.0, 100.0)), 4)


class XGBoostRealEvaluator:
    """
    Real evaluator that trains an XGBoost classifier on the Breast Cancer dataset.
    Uses 3-fold cross validation.
    """

    def __init__(self):
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import StratifiedKFold
        import xgboost as xgb
        
        self.space = get_xgboost_search_space()
        
        data = load_breast_cancer()
        self.X = data.data
        self.y = data.target
        self.cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # Suppress XGBoost warnings for clean output
        warnings.filterwarnings('ignore', category=UserWarning)

    def evaluate(self, individual):
        import xgboost as xgb
        from sklearn.model_selection import cross_val_score
        
        vec = np.copy(individual.vector)
        space = self.space

        # Map [0,1] vector → valid ConfigSpace vector for categoricals
        for i, hp_name in enumerate(list(space.keys())):
            hp = space.get_hyperparameter(hp_name)
            if isinstance(hp, CSH.CategoricalHyperparameter):
                n = len(hp.choices)
                idx = min(int(np.floor(vec[i] * n)), n - 1)
                vec[i] = float(idx)

        try:
            config = Configuration(space, vector=vec)
        except Exception:
            return 0.0  # Invalid config

        # Map to XGBoost params
        xgb_params = {
            "learning_rate": config["learning_rate"],
            "n_estimators": config["n_estimators"],
            "max_depth": config["max_depth"],
            "subsample": config["subsample"],
            "booster": config["booster"],
            "n_jobs": 1,
            "random_state": 42,
            "eval_metric": "logloss"
        }

        model = xgb.XGBClassifier(**xgb_params)
        
        try:
            scores = cross_val_score(model, self.X, self.y, cv=self.cv, scoring='accuracy', n_jobs=1)
            acc = scores.mean() * 100.0
            return round(float(acc), 4)
        except Exception:
            # If training fails (e.g. invalid dart params), return 0
            return 0.0
