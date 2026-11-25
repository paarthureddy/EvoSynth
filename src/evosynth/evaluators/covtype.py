"""
Forest Covertype Real-World Dataset Evaluator
==============================================
Evaluates hyperparameter configurations across XGBoost, LightGBM, and RandomForest
on the UCI Forest Covertype dataset (581,012 samples, 54 features, 7 classes).

Uses a stratified 20,000-sample representative slice for high-throughput search
with 2-Fold Stratified Cross-Validation and Early Stopping.

Citation: Blackard, J.A. & Dean, D.J. (1999). Forest Covertype dataset.
          UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/covertype
"""
import numpy as np
import warnings
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import Configuration
from evosynth.core.search_space import get_xgboost_search_space
from evosynth.features.generic import GenericFeatureEngineer

warnings.filterwarnings("ignore")


class CovertypeEvaluator:
    """
    Real-world multiclass classification evaluator on the UCI Forest Covertype dataset.
    Dynamically switches between XGBoost, LightGBM, and RandomForest based on the
    latent vector's model_type dimension.
    """

    def __init__(self, subsample_size=20000):
        from sklearn.datasets import fetch_covtype
        from sklearn.model_selection import train_test_split

        self.space = get_xgboost_search_space(include_model_type=True, include_fe_strategy=True)

        print(f"  [DATASET] Loading UCI Forest Covertype (581,012 samples, 54 features, 7 classes)...")
        import os
        from evosynth.paths import DATA_DIR
        _data_home = str(DATA_DIR)
        data = fetch_covtype(data_home=_data_home)
        X_full = data.data.astype(np.float32)
        y_full = data.target.astype(int) - 1  # 0-indexed: [0, 6] for XGBoost/LightGBM compatibility

        # Stratified subsample for tractable HPO search
        print(f"  [DATASET] Stratified subsampling to {subsample_size:,} rows for search phase...")
        self.X, _, self.y, _ = train_test_split(
            X_full, y_full,
            train_size=subsample_size,
            stratify=y_full,
            random_state=42
        )
        print(f"  [DATASET] Ready: {self.X.shape[0]:,} samples, {self.X.shape[1]} features, "
              f"{len(np.unique(self.y))} classes")
        print(f"  [DATASET] Class distribution: {dict(zip(*np.unique(self.y, return_counts=True)))}")

    def evaluate(self, individual):
        """
        Decodes the [0, 1]^d latent vector into hyperparameters, trains the selected
        model architecture with early stopping, and returns multiclass accuracy (%).
        """
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold

        valid_vec = np.copy(individual.vector)
        for i, hp_name in enumerate(list(self.space.keys())):
            hp = self.space.get_hyperparameter(hp_name)
            if isinstance(hp, CSH.CategoricalHyperparameter):
                num_choices = len(hp.choices)
                idx = int(np.floor(valid_vec[i] * num_choices))
                idx = min(idx, num_choices - 1)
                valid_vec[i] = float(idx)

        config = Configuration(self.space, vector=valid_vec)

        model_type = config.get("model_type", "XGBoost")
        fe_strategy = config.get("fe_strategy", "none")
        lr = config.get("learning_rate", 0.1)
        n_est = config.get("n_estimators", 100)
        depth = config.get("max_depth", 6)
        sub = config.get("subsample", 0.8)
        booster = config.get("booster", "gbtree")
        colsample = config.get("colsample_bytree", 0.8)
        reg_alpha = config.get("reg_alpha", 0.0)

        # 2-Fold Stratified CV for speed on large subsample
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        fold_scores = []

        fe = GenericFeatureEngineer(strategy=fe_strategy)

        for train_idx, val_idx in skf.split(self.X, self.y):
            X_tr, X_va = self.X[train_idx], self.X[val_idx]
            y_tr, y_va = self.y[train_idx], self.y[val_idx]

            if fe_strategy != "none":
                X_tr = fe.fit_transform(X_tr)
                X_va = fe.transform(X_va)

            try:
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
                        eval_metric="mlogloss",
                        random_state=42,
                        n_jobs=1,
                        use_label_encoder=False
                    )
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                    preds = model.predict(X_va)
                    fold_scores.append(float(np.mean(preds == y_va)))

                elif model_type == "LightGBM":
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
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=callbacks)
                    preds = model.predict(X_va)
                    fold_scores.append(float(np.mean(preds == y_va)))

                elif model_type == "ExtraTrees":
                    from sklearn.ensemble import ExtraTreesClassifier
                    et_depth = depth * 3 if depth < 10 else None
                    model = ExtraTreesClassifier(
                        n_estimators=min(n_est, 150),
                        max_depth=et_depth,
                        max_features=colsample,
                        random_state=42,
                        n_jobs=1
                    )
                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_va)
                    fold_scores.append(float(np.mean(preds == y_va)))

                else:  # RandomForest
                    rf_depth = depth * 3 if depth < 10 else None
                    model = RandomForestClassifier(
                        n_estimators=min(n_est, 150),
                        max_depth=rf_depth,
                        max_features=colsample,
                        random_state=42,
                        n_jobs=1
                    )
                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_va)
                    fold_scores.append(float(np.mean(preds == y_va)))

            except Exception:
                # Fallback: return a low score for invalid configs
                fold_scores.append(0.5)

        return float(np.mean(fold_scores)) * 100.0
