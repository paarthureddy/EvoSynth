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
            eval_metric=xgb_eval_metric, random_state=rs, n_jobs=jobs, use_label_encoder=False)
    # refit (booster is not passed at refit time, as in the original code; see fix plan)
    return xgb.XGBClassifier(
        learning_rate=params.get("learning_rate", 0.1), n_estimators=max(n_est, s.refit.min_trees_xgb),
        max_depth=params.get("max_depth", 8), subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8), reg_alpha=params.get("reg_alpha", 0.0),
        random_state=rs, n_jobs=jobs, eval_metric=xgb_eval_metric)


def fit_model(model, model_type, X_tr, y_tr, X_va=None, y_va=None, phase="search", settings=None):
    """Fit `model`. In the search phase boosted models early-stop on (X_va, y_va)."""
    if phase != "search":
        model.fit(X_tr, y_tr)
        return model
    s = settings or get_settings()
    if model_type == "XGBoost":
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    elif model_type == "LightGBM":
        import lightgbm as lgb
        callbacks = [lgb.early_stopping(stopping_rounds=s.evaluation.early_stopping_rounds, verbose=False)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=callbacks)
    else:
        model.fit(X_tr, y_tr)
    return model
