"""
The ONE place where tabular models are built from decoded hyperparameters.

phase = "search"           model used while scoring a candidate (cheap: tree cap,
                           early stopping on boosted models, n_jobs from [evaluation])
phase = "refit"            winner retrained on the refit split (min tree counts from [refit])
phase = "baseline_default" library-default model of the chosen family with
                           [experiments.baselines].refit_trees trees (Random Search / Optuna
                           refit as currently implemented; Phase 4.4 replaces it with "refit")

Settings: [evaluation], [refit], [experiments.baselines], [run].seed,
[search_space.tabular].max_depth.high, and the dataset's xgb_eval_metric.
"""
import warnings

import numpy as np

from evosynth.config import get_settings


def _bagging_depth(params, s):
    """RF/ET depth = max_depth * multiplier, unlimited when max_depth is at its upper bound."""
    depth = params["max_depth"]
    top = s.search_space.tabular.max_depth["high"]
    return depth * s.evaluation.bagging_depth_multiplier if depth < top else None


def build_model(params, phase="search", xgb_eval_metric="mlogloss", settings=None):
    """Return an unfitted classifier for params['model_type'] in the given phase."""
    s = settings or get_settings()
    m = params.get("model_type", "XGBoost")
    rs = s.run.seed

    if phase == "baseline_default":
        n = s.experiments.baselines.refit_trees
        if m == "ExtraTrees":
            from sklearn.ensemble import ExtraTreesClassifier
            return ExtraTreesClassifier(n_estimators=n, random_state=rs, n_jobs=-1)
        if m == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=n, random_state=rs, n_jobs=-1)
        if m == "LightGBM":
            import lightgbm as lgb
            return lgb.LGBMClassifier(n_estimators=n, random_state=rs, n_jobs=-1, verbose=-1)
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=n, random_state=rs, n_jobs=-1, eval_metric=xgb_eval_metric)

    if phase not in ("search", "refit"):
        raise ValueError(f"unknown phase '{phase}'")
    search = phase == "search"
    jobs = s.evaluation.model_n_jobs if search else s.refit.model_n_jobs
    n_est = params.get("n_estimators", 100 if search else 150)

    if m in ("ExtraTrees", "RandomForest"):
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        cls = ExtraTreesClassifier if m == "ExtraTrees" else RandomForestClassifier
        trees = min(n_est, s.evaluation.search_tree_cap) if search else max(n_est, s.refit.min_trees_bagging)
        return cls(n_estimators=trees, max_depth=_bagging_depth(params, s),
                   max_features=params.get("colsample_bytree", 0.8), random_state=rs, n_jobs=jobs)

    if m == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            learning_rate=params.get("learning_rate", 0.1),
            n_estimators=n_est if search else max(n_est, s.refit.min_trees_lgbm),
            max_depth=params.get("max_depth", 6), subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8), reg_alpha=params.get("reg_alpha", 0.0),
            random_state=rs, n_jobs=jobs, verbose=-1)

    import xgboost as xgb
    if search:
        return xgb.XGBClassifier(
            learning_rate=params.get("learning_rate", 0.1), n_estimators=n_est,
            max_depth=params.get("max_depth", 6), subsample=params.get("subsample", 0.8),
            booster=params.get("booster", "gbtree"), colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_alpha=params.get("reg_alpha", 0.0), early_stopping_rounds=s.evaluation.early_stopping_rounds,
            eval_metric=xgb_eval_metric, random_state=rs, n_jobs=jobs)
    # refit (booster is not passed at refit time, as in the original code; see fix plan)
    return xgb.XGBClassifier(
        learning_rate=params.get("learning_rate", 0.1), n_estimators=max(n_est, s.refit.min_trees_xgb),
        max_depth=params.get("max_depth", 8), subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8), reg_alpha=params.get("reg_alpha", 0.0),
        random_state=rs, n_jobs=jobs, eval_metric=xgb_eval_metric)


class _Relabeled:
    """
    XGBoost requires the training labels to be exactly 0..K-1. When a training sample lacks a
    class (rare classes, e.g. Poker Hand), the model is fitted on re-coded labels and this
    wrapper maps predictions back to the dataset's labels.
    """

    def __init__(self, model, classes):
        self.model, self.classes_ = model, np.asarray(classes)

    def predict(self, X):
        return self.classes_[np.asarray(self.model.predict(X)).astype(int)]

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def fit_model(model, model_type, X_tr, y_tr, X_es=None, y_es=None, phase="search", settings=None):
    """
    Fit `model` and return the fitted model (use the return value: it may be a _Relabeled wrapper).
    In the search phase boosted models early-stop on (X_es, y_es), a split of the TRAINING fold
    made by the caller (Phase 3.3); the scoring fold is never used for early stopping.
    """
    s = settings or get_settings()
    classes = np.unique(y_tr)
    relabel = model_type == "XGBoost" and not np.array_equal(classes, np.arange(len(classes)))
    if relabel:
        code = {c: i for i, c in enumerate(classes)}
        y_tr = np.array([code[c] for c in y_tr])
        if y_es is not None:
            keep = np.isin(y_es, classes)
            X_es, y_es = X_es[keep], np.array([code[c] for c in y_es[keep]])

    if model_type == "XGBoost" and getattr(model, "eval_metric", None) in ("logloss", "mlogloss"):
        # XGBoost picks a binary or multiclass objective from the labels it sees; the loss must match
        model.set_params(eval_metric="logloss" if len(classes) <= 2 else "mlogloss")

    early_stop = phase == "search" and X_es is not None and len(y_es) > 0
    if model_type == "XGBoost" and early_stop:
        model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
    elif model_type == "XGBoost" and phase == "search":
        model.set_params(early_stopping_rounds=None)
        model.fit(X_tr, y_tr, verbose=False)
    elif model_type == "LightGBM" and early_stop:
        import lightgbm as lgb
        callbacks = [lgb.early_stopping(stopping_rounds=s.evaluation.early_stopping_rounds, verbose=False)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], callbacks=callbacks)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)
    return _Relabeled(model, classes) if relabel else model
