"""
TabularEvaluator: scores a latent vector on a registered tabular dataset.

decode vector -> stratified k-fold CV on the dataset's search subsample ->
per fold: fit the FE strategy on train, build the model (models.factory, phase="search"),
fit with early stopping, accuracy on the validation fold -> mean accuracy in %.

Settings: [evaluation] (cv_folds, failed_fold_score, ...), [datasets.<name>]
(search_subsample, xgb_eval_metric, display_name), [run].data_seed.
"""
import warnings

import numpy as np

from evosynth import datasets
from evosynth.config import get_settings
from evosynth.core.decoding import decode_vector_to_dict
from evosynth.features.generic import GenericFeatureEngineer
from evosynth.models.factory import build_model, fit_model
from evosynth.search_spaces.tabular import get_tabular_search_space

warnings.filterwarnings("ignore")


class TabularEvaluator:
    """
    Real-world tabular classification evaluator. Switches model family (XGBoost,
    LightGBM, RandomForest, ExtraTrees) and feature engineering from the latent vector.
    """

    def __init__(self, dataset="covertype", subsample_size=None, settings=None):
        self.settings = settings or get_settings()
        self.dataset = dataset
        self.spec = datasets.get_spec(dataset, self.settings)
        n = self.spec.search_subsample if subsample_size is None else subsample_size
        self.space = get_tabular_search_space(include_model_type=True, include_fe_strategy=True,
                                              settings=self.settings)

        print(f"  [DATASET] Loading {self.spec.display_name}...")
        X_full, y_full = datasets.load(dataset, self.settings)
        print(f"  [DATASET] Stratified subsampling to {n:,} rows for search phase...")
        self.X, self.y = datasets.search_subsample(X_full, y_full, n, self.settings.run.data_seed)
        print(f"  [DATASET] Ready: {self.X.shape[0]:,} samples, {self.X.shape[1]} features, "
              f"{len(np.unique(self.y))} classes")
        print(f"  [DATASET] Class distribution: {dict(zip(*np.unique(self.y, return_counts=True)))}")

    def evaluate(self, individual):
        """Decode the latent vector, cross-validate the configuration, return accuracy (%)."""
        from sklearn.model_selection import StratifiedKFold

        s = self.settings
        params = decode_vector_to_dict(individual.vector, self.space)
        model_type = params.get("model_type", "XGBoost")
        fe_strategy = params.get("fe_strategy", "none")

        skf = StratifiedKFold(n_splits=s.evaluation.cv_folds, shuffle=True, random_state=s.run.data_seed)
        fe = GenericFeatureEngineer(strategy=fe_strategy)
        fold_scores = []
        for train_idx, val_idx in skf.split(self.X, self.y):
            X_tr, X_va = self.X[train_idx], self.X[val_idx]
            y_tr, y_va = self.y[train_idx], self.y[val_idx]
            if fe_strategy != "none":
                X_tr = fe.fit_transform(X_tr)
                X_va = fe.transform(X_va)
            try:
                model = build_model(params, phase="search", xgb_eval_metric=self.spec.xgb_eval_metric, settings=s)
                fit_model(model, model_type, X_tr, y_tr, X_va, y_va, phase="search", settings=s)
                fold_scores.append(float(np.mean(model.predict(X_va) == y_va)))
            except Exception:
                # Invalid configuration: record the configured fallback score for this fold
                fold_scores.append(s.evaluation.failed_fold_score)
        return float(np.mean(fold_scores)) * 100.0
