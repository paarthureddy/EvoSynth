"""
Post-search generalisation refit, shared by every experiment:
load dataset -> stratified refit split ([refit] train_rows / test_rows) -> fit the
feature-engineering strategy on train only -> build model -> fit -> test score.
"""
import numpy as np

from evosynth.config import get_settings
from evosynth import datasets
from evosynth.features.generic import GenericFeatureEngineer
from evosynth.models.factory import build_model, fit_model


def refit_split(dataset="covertype", settings=None):
    """(X_tr, X_te, y_tr, y_te) for the refit; identical rows for every method."""
    s = settings or get_settings()
    X, y = datasets.load(dataset, s)
    return datasets.refit_split(X, y, s.refit.train_rows, s.refit.test_rows, s.run.data_seed)


def refit_and_score(params, dataset="covertype", phase="refit", verbose=True, settings=None):
    """
    Retrain the configuration `params` (decoded hyperparameters) on the refit train rows
    and return a dict with test accuracy (%), model type, FE strategy and the fitted model.
    """
    s = settings or get_settings()
    spec = datasets.get_spec(dataset, s)
    X_tr, X_te, y_tr, y_te = refit_split(dataset, s)

    model_type = params.get("model_type", "XGBoost")
    fe_strategy = params.get("fe_strategy", "none")
    fe = GenericFeatureEngineer(strategy=fe_strategy)
    if fe_strategy != "none":
        if verbose:
            print(f"  [FE] Applying winning feature engineering strategy: '{fe_strategy}'")
        X_tr = fe.fit_transform(X_tr)
        X_te = fe.transform(X_te)
    if verbose:
        print(f"  [MODEL] Training winning architecture: '{model_type}' on {s.refit.train_rows:,} samples...")

    model = build_model(params, phase=phase, xgb_eval_metric=spec.xgb_eval_metric, settings=s)
    fit_model(model, model_type, X_tr, y_tr, phase=phase, settings=s)
    test_acc = float(np.mean(model.predict(X_te) == y_te)) * 100.0
    return {"test_accuracy": test_acc, "model_type": model_type, "fe_strategy": fe_strategy, "model": model}
