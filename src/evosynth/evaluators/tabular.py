"""
TabularEvaluator: scores a latent vector on a registered tabular dataset.

decode vector -> stratified k-fold CV on the dataset's search rows (datasets.partition) ->
per fold: fit the FE strategy on the training fold, hold back an early-stopping set from the
training fold (boosted models only), build the model (models.factory, phase="search"), fit,
score the validation fold with the dataset's metric -> mean score in %.

Settings: [evaluation] (cv_folds, early_stopping_fraction, failed_fold_score, ...),
[datasets.<name>] (metric, search_subsample, xgb_eval_metric, display_name), [run].data_seed.
"""
import threading
import warnings

import numpy as np

from evosynth import datasets
from evosynth.config import get_settings
from evosynth.core.decoding import decode_vector_to_dict
from evosynth.datasets.splits import _stratify
from evosynth.features.generic import GenericFeatureEngineer
from evosynth.models.factory import build_model, fit_model
from evosynth.models.metrics import score
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
        self.spec = datasets.get_spec(dataset, self.settings, kind="tabular")
        self.metric = self.spec.metric
        self.space = get_tabular_search_space(include_model_type=True, include_fe_strategy=True,
                                              settings=self.settings)
        self.failed_folds = 0
        self._lock = threading.Lock()

        print(f"  [DATASET] Loading {self.spec.display_name}...")
        X, y = datasets.partition(dataset, self.settings)["search"]
        if subsample_size is not None and subsample_size < len(y):
            X, y = datasets.sample(X, y, subsample_size, self.settings.run.data_seed)
        self.X, self.y = X, y
        print(f"  [DATASET] Search rows: {self.X.shape[0]:,} (stratified; the {self.spec.test_rows:,} "
              f"test rows are held out), {self.X.shape[1]} features, {len(np.unique(self.y))} classes, "
              f"metric: {self.metric}")
        print(f"  [DATASET] Class distribution: {dict(zip(*np.unique(self.y, return_counts=True)))}")

    def evaluate(self, individual):
        """Decode the latent vector, cross-validate the configuration, return the score (%)."""
        params = decode_vector_to_dict(individual.vector, self.space)
        return self._cv_score(params, self.settings.evaluation.cv_folds)

    def evaluate_config(self, params, folds):
        """Score decoded hyperparameters with `folds`-fold CV (used by confirm-before-select)."""
        return self._cv_score(params, folds)

    # ------------------------------------------------------------------------------
    def _cv_score(self, params, folds):
        from sklearn.model_selection import StratifiedKFold

        s = self.settings
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=s.run.data_seed)
        fold_scores = []
        for i, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y), 1):
            try:
                fold_scores.append(self._fit_and_score(params, train_idx, val_idx, fold_seed=i))
            except Exception as e:
                # Invalid configuration (Phase 3.4): score failed_fold_score (0) and say so
                with self._lock:
                    self.failed_folds += 1
                print(f"  [EVAL] fold {i}/{folds} failed for {params.get('model_type')} "
                      f"({type(e).__name__}: {str(e)[:120]}); scored {s.evaluation.failed_fold_score}")
                fold_scores.append(s.evaluation.failed_fold_score)
        return float(np.mean(fold_scores)) * 100.0

    def _fit_and_score(self, params, train_idx, val_idx, fold_seed):
        from sklearn.model_selection import train_test_split

        s = self.settings
        model_type = params.get("model_type", "XGBoost")
        fe_strategy = params.get("fe_strategy", "none")
        X_tr, X_va = self.X[train_idx], self.X[val_idx]
        y_tr, y_va = self.y[train_idx], self.y[val_idx]
        if fe_strategy != "none":
            fe = GenericFeatureEngineer(strategy=fe_strategy)
            X_tr = fe.fit_transform(X_tr)
            X_va = fe.transform(X_va)

        X_es = y_es = None
        if model_type in ("XGBoost", "LightGBM"):
            # Phase 3.3: early stopping on a split of the TRAINING fold, never on the scoring fold
            X_tr, X_es, y_tr, y_es = train_test_split(
                X_tr, y_tr, test_size=s.evaluation.early_stopping_fraction, stratify=_stratify(y_tr),
                random_state=s.run.data_seed + fold_seed)

        model = build_model(params, phase="search", xgb_eval_metric=self.spec.xgb_eval_metric, settings=s)
        model = fit_model(model, model_type, X_tr, y_tr, X_es, y_es, phase="search", settings=s)
        return score(model, X_va, y_va, self.metric)
