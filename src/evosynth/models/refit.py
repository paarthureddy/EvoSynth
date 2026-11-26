"""
Post-search generalisation refit, shared by EVERY tuned method (EvoSynth, Random Search, Optuna;
Phase 4.4), so they differ only in which configuration they selected:

dataset -> datasets.partition (train / test rows from [datasets.<name>], test rows never seen by
any search) -> fit the feature-engineering strategy on train only -> build model (phase="refit")
-> boosted models early-stop on a split of the train rows ([evaluation] early_stopping_fraction),
as in the search -> test score with the dataset's metric -> inference cost (models/inference.py).
"""
from sklearn.model_selection import train_test_split

from evosynth.config import get_settings
from evosynth import datasets
from evosynth.datasets.splits import _stratify
from evosynth.features.generic import GenericFeatureEngineer
from evosynth.models.factory import build_model, fit_model
from evosynth.models.inference import measure
from evosynth.models.metrics import score


def refit_split(dataset="covertype", settings=None):
    """(X_tr, X_te, y_tr, y_te) for the refit; identical rows for every method."""
    parts = datasets.partition(dataset, settings or get_settings())
    (X_tr, y_tr), (X_te, y_te) = parts["train"], parts["test"]
    return X_tr, X_te, y_tr, y_te


def refit_and_score(params, dataset="covertype", verbose=True, settings=None, model_path=None):
    """
    Retrain the configuration `params` (decoded hyperparameters) on the refit train rows.
    Returns a dict with the test score (%), metric, model type, FE strategy, row counts,
    inference cost and the fitted model. With model_path, the fitted model is pickled there.
    """
    s = settings or get_settings()
    spec = datasets.get_spec(dataset, s, kind="tabular")
    X_tr, X_te, y_tr, y_te = refit_split(dataset, s)
    n_train = len(y_tr)

    model_type = params.get("model_type", "XGBoost")
    fe_strategy = params.get("fe_strategy", "none")
    fe = GenericFeatureEngineer(strategy=fe_strategy)
    if fe_strategy != "none":
        if verbose:
            print(f"  [FE] Applying the selected feature engineering strategy: '{fe_strategy}'")
        X_tr = fe.fit_transform(X_tr)
        X_te = fe.transform(X_te)

    X_es = y_es = None
    if model_type in ("XGBoost", "LightGBM") and s.evaluation.early_stopping:
        X_tr, X_es, y_tr, y_es = train_test_split(X_tr, y_tr, test_size=s.evaluation.early_stopping_fraction,
                                                  stratify=_stratify(y_tr), random_state=s.run.data_seed)
    if verbose:
        print(f"  [MODEL] Training '{model_type}' on {n_train:,} rows"
              + (f" ({len(y_es):,} of them held back for early stopping)" if y_es is not None else "") + "...")

    model = build_model(params, phase="refit", xgb_eval_metric=spec.xgb_eval_metric, settings=s)
    model = fit_model(model, model_type, X_tr, y_tr, X_es, y_es, phase="refit", settings=s)
    test_score = score(model, X_te, y_te, spec.metric) * 100.0
    inference = measure(model, X_te, s)
    if model_path is not None:
        import pickle
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "feature_engineering": fe, "config": params}, f)
    return {"test_score": test_score, "metric": spec.metric, "model_type": model_type,
            "fe_strategy": fe_strategy, "train_rows": n_train, "test_rows": len(y_te),
            "inference": inference, "model": model}
