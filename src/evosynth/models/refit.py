"""
Post-search generalisation refit, shared by every experiment:
dataset -> datasets.partition (train / test rows from [datasets.<name>], test rows never seen
by any search) -> fit the feature-engineering strategy on train only -> build model -> fit ->
test score with the dataset's metric.
"""
from evosynth.config import get_settings
from evosynth import datasets
from evosynth.features.generic import GenericFeatureEngineer
from evosynth.models.factory import build_model, fit_model
from evosynth.models.metrics import score


def refit_split(dataset="covertype", settings=None):
    """(X_tr, X_te, y_tr, y_te) for the refit; identical rows for every method."""
    parts = datasets.partition(dataset, settings or get_settings())
    (X_tr, y_tr), (X_te, y_te) = parts["train"], parts["test"]
    return X_tr, X_te, y_tr, y_te


def refit_and_score(params, dataset="covertype", phase="refit", verbose=True, settings=None):
    """
    Retrain the configuration `params` (decoded hyperparameters) on the refit train rows
    and return a dict with the test score (%), metric, model type, FE strategy and the fitted model.
    """
    s = settings or get_settings()
    spec = datasets.get_spec(dataset, s, kind="tabular")
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
        print(f"  [MODEL] Training winning architecture: '{model_type}' on {len(y_tr):,} samples...")

    model = build_model(params, phase=phase, xgb_eval_metric=spec.xgb_eval_metric, settings=s)
    model = fit_model(model, model_type, X_tr, y_tr, phase=phase, settings=s)
    test_score = score(model, X_te, y_te, spec.metric) * 100.0
    return {"test_score": test_score, "metric": spec.metric, "model_type": model_type,
            "fe_strategy": fe_strategy, "train_rows": len(y_tr), "test_rows": len(y_te), "model": model}
