"""Phase 3: dataset registry and splits, metrics, early-stopping split, failed folds, confirm-before-select."""
import numpy as np
import pandas as pd
import pytest

from evosynth import datasets
from evosynth.config import ConfigError, load_config, set_settings
from evosynth.core.individual import Individual
from evosynth.datasets import loaders, registry
from evosynth.experiments.selection import confirm_best
from evosynth.models import factory
from evosynth.models.factory import build_model, fit_model
from evosynth.models.metrics import score
from evosynth.search_spaces.tabular import get_tabular_search_space

SMALL = ["datasets.covertype.test_rows=200", "datasets.covertype.search_subsample=300",
         "datasets.covertype.refit_train_rows=400"]


@pytest.fixture
def fake_covertype(monkeypatch):
    """Covertype replaced by 1,000 synthetic rows (3 classes) so no download is needed."""
    rng = np.random.default_rng(0)
    X = rng.random((1000, 6)).astype(np.float32)
    y = (X[:, 0] * 3).astype(int).clip(0, 2)
    X[:, 1] = np.arange(1000)                      # row id, to check which rows go where
    monkeypatch.setitem(registry._CACHE, "covertype", (X, y))
    monkeypatch.setattr(registry, "_PARTS", {})
    s = load_config(overrides=SMALL)
    set_settings(s)
    return s


# ---------------------------------------------------------------- splits
def test_test_rows_are_never_in_search_or_train(fake_covertype):
    parts = datasets.partition("covertype")
    ids = {k: set(v[0][:, 1].astype(int)) for k, v in parts.items()}
    assert [len(parts[k][1]) for k in ("test", "search", "train")] == [200, 300, 400]
    assert not ids["test"] & ids["search"] and not ids["test"] & ids["train"]


def test_partition_is_reproducible_and_follows_data_seed(fake_covertype, monkeypatch):
    a = datasets.partition("covertype")["test"][0][:, 1]
    monkeypatch.setattr(registry, "_PARTS", {})
    assert np.array_equal(a, datasets.partition("covertype")["test"][0][:, 1])
    set_settings(load_config(overrides=SMALL + ["run.data_seed=7"]))
    assert not np.array_equal(a, datasets.partition("covertype")["test"][0][:, 1])


def test_too_large_splits_are_rejected(fake_covertype):
    set_settings(load_config(overrides=["datasets.covertype.test_rows=900"]))
    with pytest.raises(ValueError, match="test_rows"):
        datasets.partition("covertype")


# ---------------------------------------------------------------- loaders
def test_openml_frame_preprocessing():
    frame = pd.DataFrame({
        "num": [1.0, np.nan, 3.0, 5.0],
        "cat": pd.Categorical(["b", "a", None, "b"]),
        "class": ["<=50K", ">50K", "<=50K", ">50K"],
    })
    X, y = loaders._frame_to_arrays(frame, "class")
    assert X.dtype == np.float32 and not np.isnan(X).any()
    assert X[1, 0] == 3.0                               # median fill
    assert list(X[:, 1]) == [1, 0, 2, 1]                # sorted categories, missing → own code
    assert list(y) == [0, 1, 0, 1]


# ---------------------------------------------------------------- metrics / models
def test_metrics():
    from sklearn.tree import DecisionTreeClassifier
    X = np.array([[0.0], [1.0], [2.0], [3.0]]); y = np.array([0, 0, 1, 1])
    m = DecisionTreeClassifier(random_state=0).fit(X, y)
    assert score(m, X, y, "accuracy") == 1.0 == score(m, X, y, "roc_auc") == score(m, X, y, "balanced_accuracy")
    with pytest.raises(ValueError):
        score(m, X, y, "f1")


def test_xgboost_has_no_use_label_encoder_and_handles_a_missing_class():
    params = {"model_type": "XGBoost", "n_estimators": 20, "learning_rate": 0.3, "max_depth": 3}
    model = build_model(params, phase="search")
    assert "use_label_encoder" not in model.get_params()
    rng = np.random.default_rng(1)
    X = rng.random((60, 3)); y = np.where(X[:, 0] > 0.5, 2, 0)      # class 1 absent
    fitted = fit_model(model, "XGBoost", X[:40], y[:40], X[40:], y[40:], phase="search")
    assert set(np.unique(fitted.predict(X))) <= {0, 2}


def test_bad_dataset_settings_are_rejected():
    with pytest.raises(ConfigError, match="metric"):
        load_config(overrides=['datasets.higgs.metric="f1"'])
    with pytest.raises(ConfigError, match="experiments.nas.dataset"):
        load_config(overrides=['experiments.nas.dataset="covertype"'])


# ---------------------------------------------------------------- evaluator
def _evaluator(fake_covertype):
    from evosynth.evaluators.tabular import TabularEvaluator
    return TabularEvaluator("covertype")


def test_failed_fold_scores_zero_and_is_counted(fake_covertype, monkeypatch, capsys):
    import evosynth.evaluators.tabular as tab
    ev = _evaluator(fake_covertype)

    def boom(*a, **k):
        raise RuntimeError("bad config")
    monkeypatch.setattr(tab, "build_model", boom)
    assert ev.evaluate(Individual(np.full(9, 0.5))) == 0.0
    assert ev.failed_folds == 2 and "fold 1/2 failed" in capsys.readouterr().out


def test_early_stopping_set_comes_from_the_training_fold(fake_covertype, monkeypatch):
    import evosynth.evaluators.tabular as tab
    ev = _evaluator(fake_covertype)
    seen = []

    def spy(model, model_type, X_tr, y_tr, X_es=None, y_es=None, **kw):
        seen.append((set(X_tr[:, 1]), set(X_es[:, 1]) if X_es is not None else set()))
        return factory.fit_model(model, model_type, X_tr, y_tr, X_es, y_es, **kw)
    monkeypatch.setattr(tab, "fit_model", spy)
    cfg = {"model_type": "LightGBM", "fe_strategy": "none", "n_estimators": 20, "learning_rate": 0.1,
           "max_depth": 3, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "booster": "gbtree"}
    ev.evaluate_config(cfg, folds=2)
    for (train_ids, es_ids), (_, val_idx) in zip(seen, _folds(ev)):
        val_ids = set(ev.X[val_idx][:, 1])
        assert es_ids and not es_ids & val_ids and not es_ids & train_ids
        assert len(es_ids) == pytest.approx(0.15 * (len(train_ids) + len(es_ids)), abs=1)


def _folds(ev):
    from sklearn.model_selection import StratifiedKFold
    return list(StratifiedKFold(2, shuffle=True, random_state=ev.settings.run.data_seed).split(ev.X, ev.y))


def test_real_fit_scores_above_chance(fake_covertype):
    ev = _evaluator(fake_covertype)
    for z in (0.0, 0.3, 0.6, 0.99):                    # XGBoost, RandomForest, LightGBM, ExtraTrees
        v = np.full(9, 0.5); v[5] = z
        assert ev.evaluate(Individual(v)) > 50.0
    assert ev.failed_folds == 0


# ---------------------------------------------------------------- confirm before select
class _FakeEvaluator:
    def __init__(self, confirmed):
        self.confirmed = confirmed
        self.calls = 0

    def evaluate_config(self, cfg, folds):
        self.calls += 1
        return self.confirmed[cfg["model_type"]]


def test_confirm_best_picks_the_best_confirmed_distinct_config():
    space = get_tabular_search_space()
    def ind(z_model, fit):
        v = np.full(9, 0.5); v[5] = z_model
        i = Individual(v); i.set_fitness(fit); i.source_algorithm = "T1_GA"
        return i
    pop = [ind(0.0, 90.0), ind(0.0, 89.0), ind(0.3, 88.0), ind(0.6, 70.0)]   # first two decode the same
    fake = _FakeEvaluator({"XGBoost": 80.0, "RandomForest": 86.0, "LightGBM": 60.0})
    set_settings(load_config(overrides=["selection.top_k=3"]))
    chosen, cfg, report = confirm_best(fake, pop, space)
    assert fake.calls == 3 and cfg["model_type"] == "RandomForest" and chosen.fitness == 88.0
    assert report["selected_search_rank"] == 2 and len(report["candidates"]) == 3
    set_settings(load_config(overrides=["selection.enabled=false"]))
    chosen, _, report = confirm_best(fake, pop, space)
    assert chosen.fitness == 90.0 and report is None
