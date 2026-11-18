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
                idx = min(idx, num_choices - 1)  # Handle exactly 1.0
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
    Evaluates hyperparameter configurations dynamically across multiple model
    architectures (XGBoost, RandomForest, LightGBM) on the Breast Cancer classification dataset.
    Incorporates Early Stopping on gradient boosted trees to eliminate wasted computation
    on unpromising hyperparameter configurations.
    """

    def __init__(self, space=None):
        from .search_space import get_xgboost_search_space
        from sklearn.datasets import load_breast_cancer
        import warnings
        warnings.filterwarnings("ignore")

        self.space = space or get_xgboost_search_space(include_model_type=True)

        # Load dataset once
        data = load_breast_cancer()
        self.X = data.data
        self.y = data.target

    def evaluate(self, individual):
        """
        Receives an Individual (with a [0, 1] scaled vector),
        decodes it into real hyperparameter values, dynamically trains
        the selected model with early stopping, and returns cross-validation accuracy.
        """
        import xgboost as xgb
        from sklearn.model_selection import StratifiedKFold
        from sklearn.ensemble import RandomForestClassifier

        valid_vec = np.copy(individual.vector)
        for i, hp_name in enumerate(list(self.space.keys())):
            hp = self.space.get_hyperparameter(hp_name)
            if isinstance(hp, CSH.CategoricalHyperparameter):
                num_choices = len(hp.choices)
                idx = int(np.floor(valid_vec[i] * num_choices))
                idx = min(idx, num_choices - 1)
                valid_vec[i] = float(idx)

        config = Configuration(self.space, vector=valid_vec)

        # Dynamic model type resolution
        model_type = config["model_type"] if "model_type" in config else "XGBoost"

        # Shared / standard hyperparameters
        lr = config.get("learning_rate", 0.1)
        n_est = config.get("n_estimators", 100)
        depth = config.get("max_depth", 6)
        sub = config.get("subsample", 0.8)
        booster = config.get("booster", "gbtree")
        colsample = config.get("colsample_bytree", 0.8)
        reg_alpha = config.get("reg_alpha", 0.0)

        # 3-Fold Stratified Cross-Validation with Early Stopping
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fold_scores = []

        for train_idx, val_idx in skf.split(self.X, self.y):
            X_tr, X_va = self.X[train_idx], self.X[val_idx]
            y_tr, y_va = self.y[train_idx], self.y[val_idx]

            if model_type == "XGBoost":
                model = xgb.XGBClassifier(
                    learning_rate=lr,
                    n_estimators=n_est,
                    max_depth=depth,
                    subsample=sub,
                    booster=booster,
                    colsample_bytree=colsample,
                    reg_alpha=reg_alpha,
                    early_stopping_rounds=10,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=1
                )
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                preds = model.predict(X_va)
                fold_scores.append(float(np.mean(preds == y_va)))

            elif model_type == "LightGBM":
                try:
                    import lightgbm as lgb
                    model = lgb.LGBMClassifier(
                        learning_rate=lr,
                        n_estimators=n_est,
                        max_depth=depth,
                        subsample=sub,
                        colsample_bytree=colsample,
                        reg_alpha=reg_alpha,
                        random_state=42,
                        n_jobs=1,
                        verbose=-1
                    )
                    callbacks = [lgb.early_stopping(stopping_rounds=10, verbose=False)]
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=callbacks)
                    preds = model.predict(X_va)
                    fold_scores.append(float(np.mean(preds == y_va)))
                except Exception:
                    # Fallback to XGBoost if LightGBM has unexpected dynamic issue
                    model = xgb.XGBClassifier(
                        learning_rate=lr, n_estimators=n_est, max_depth=depth,
                        early_stopping_rounds=10, eval_metric="logloss", random_state=42, n_jobs=1
                    )
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                    preds = model.predict(X_va)
                    fold_scores.append(float(np.mean(preds == y_va)))

            else:  # RandomForest
                model = RandomForestClassifier(
                    n_estimators=min(n_est, 150),
                    max_depth=depth,
                    max_features=colsample,
                    random_state=42,
                    n_jobs=1
                )
                model.fit(X_tr, y_tr)
                preds = model.predict(X_va)
                fold_scores.append(float(np.mean(preds == y_va)))

        # Return accuracy as percentage (0-100)
        return float(np.mean(fold_scores)) * 100.0


# Backward compatible and semantic aliases
ModelSwitchingEvaluator = XGBoostEvaluator

