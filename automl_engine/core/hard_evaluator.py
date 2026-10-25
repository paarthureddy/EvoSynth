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

        model_type = config["model_type"] if "model_type" in config else "XGBoost"
        lr = config.get("learning_rate", 0.1)
        n_est = config.get("n_estimators", 100)
        depth = config.get("max_depth", 6)
        sub = config.get("subsample", 0.8)
        booster = config.get("booster", "gbtree")
        colsample = config.get("colsample_bytree", 0.8)
        reg_alpha = config.get("reg_alpha", 0.0)

        from sklearn.model_selection import StratifiedKFold
        from sklearn.ensemble import RandomForestClassifier

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

        return float(np.mean(fold_scores)) * 100.0

