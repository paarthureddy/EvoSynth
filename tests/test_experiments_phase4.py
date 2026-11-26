"""Phase 4: ablation (real metrics, joint passive scoring, resume, conclusion), shared refit,
defaults baseline, inference measurement, summary tables, controller records."""
import json
import os
import pickle

import numpy as np
import pytest

from evosynth.config import get_settings, load_config, set_settings, with_overrides
from evosynth.datasets import registry
from evosynth.engine.controller import DynamicOptimizer
from evosynth.engine.scoring import CompositeScoringEngine
from evosynth.experiments import ablation
from evosynth.experiments.run_context import RunContext
from evosynth.experiments.summary import build_summary

SMALL = ["datasets.covertype.test_rows=200", "datasets.covertype.search_subsample=300",
         "datasets.covertype.refit_train_rows=400"]


@pytest.fixture
def fake_covertype(monkeypatch):
    rng = np.random.default_rng(0)
    X = rng.random((1000, 6)).astype(np.float32)
    y = (X[:, 0] * 3).astype(int).clip(0, 2)
    monkeypatch.setitem(registry._CACHE, "covertype", (X, y))
    monkeypatch.setattr(registry, "_PARTS", {})
    set_settings(load_config(overrides=SMALL))


class ToyEvaluator:
    """The toy objective dressed as a dataset evaluator (what the ablation needs)."""
    def __init__(self, toy):
        self.inner, self.metric, self.dataset, self.calls = toy, "accuracy", "toy", 0

    def evaluate(self, ind):
        self.calls += 1
        return self.inner.evaluate(ind)


# ---------------------------------------------------------------- controller records
def test_controller_records_scoring_calls_and_decisions(toy_problem):
    space, ev = toy_problem
    opt = DynamicOptimizer(space, ev, budget=200, pop_size=40, t1_iters=2, k_iters=1, max_workers=1)
    opt.run()
    assert len(opt.scoring_history) >= 3
    first = opt.scoring_history[0]
    assert set(first["islands"]) == {"T1_GA", "T1_PSO", "T1_DE", "T1_CMAES"}
    assert {"fitness", "diversity", "composite", "ucb"} <= set(first["islands"]["T1_GA"])
    assert any(d["action"] == "Promotion" for d in opt.decisions)


# ---------------------------------------------------------------- passive scoring
def test_passive_scores_use_the_controller_engine():
    log = lambda best, div, sec: [{"evals": 40, "best": best[0], "diversity": div, "eval_seconds": sec[0]},
                                  {"evals": 80, "best": best[1], "diversity": div, "eval_seconds": sec[1]}]
    iso = {"GA": {"log": log((50, 60), 0.4, (1, 2))}, "PSO": {"log": log((55, 56), 0.2, (1, 3))}}
    out = ablation.passive_scores(iso, [40, 80])
    raw = {a: {k: out[80][a][k] for k in ("fitness", "speed", "efficiency", "diversity", "trend")} for a in iso}
    assert out[80]["GA"]["speed"] == 10 and out[80]["PSO"]["speed"] == 1
    assert out[80]["GA"]["efficiency"] == pytest.approx(10.0) and out[80]["PSO"]["efficiency"] == pytest.approx(0.5)
    expected = CompositeScoringEngine().calculate_scores(raw)
    assert out[80]["GA"]["composite"] == pytest.approx(expected["GA"])
    assert out[80]["GA"]["ucb"] - out[80]["GA"]["composite"] == pytest.approx(
        out[80]["PSO"]["ucb"] - out[80]["PSO"]["composite"])          # same bonus: same evaluations used


# ---------------------------------------------------------------- ablation end to end
def _ablation(tmp_path, toy_problem, splits=None):
    space, toy = toy_problem
    ev = ToyEvaluator(toy)
    with RunContext("ablation", "toy", "T-160", results_dir=tmp_path, resume=True) as run:
        methods = ablation.run_benchmark_setup(run, "T-160", budget=160, pop_size=40, evaluator=ev, space=space,
                                               max_workers=1, checkpoint_eval=40, checkpoint_interval=10,
                                               splits=splits)
    return methods, ev, run.path


def test_ablation_reports_measured_values_and_a_generated_conclusion(tmp_path, toy_problem):
    methods, ev, path = _ablation(tmp_path, toy_problem, splits=["stratified", "elite"])
    assert set(methods) == {"GA", "PSO", "DE", "CMAES", "EvoSynth", "EvoSynth[elite]"}
    assert ev.calls == 6 * 160
    evo = methods["EvoSynth"]
    for k in ("diversity_at_mark", "composite_at_mark", "ucb_at_mark", "final_diversity", "final_ucb"):
        assert evo[k] is not None and evo[k] not in (0.5, 0.45, 0.625, 0.654, 0.704, 0.72)   # no v1 constants
    assert evo["decisions"].get("Promotion") == 1
    for a in ("GA", "PSO", "DE", "CMAES"):
        assert 0.0 <= methods[a]["final_composite"] <= 1.0 and methods[a]["total_evals"] == 160
    result = json.load(open(os.path.join(path, "result.json"), encoding="utf-8"))
    assert result["conclusion"][0].startswith("At the full budget (160 evaluations) EvoSynth")
    assert f"{methods['EvoSynth']['best_cv_score']:.2f}%" in result["conclusion"][0]
    assert os.path.exists(os.path.join(path, "population_elite.csv"))


def test_ablation_resumes_with_full_optimizer_state(tmp_path, toy_problem):
    first, _, path = _ablation(tmp_path, toy_problem)
    ckpt = os.path.join(path, "checkpoints")
    for a in ("GA", "PSO", "DE", "CMAES"):                  # every optimizer's state survives pickling
        st = pickle.load(open(os.path.join(ckpt, f"{a}_state.pkl"), "rb"))
        assert st["evals"] == 160 and type(st["optimizer"]).__name__
    again, ev, _ = _ablation(tmp_path, toy_problem)
    assert ev.calls == 0                                    # everything came from the checkpoints
    assert {k: v["best_cv_score"] for k, v in again.items()} == {k: v["best_cv_score"] for k, v in first.items()}


def test_interrupted_isolated_run_continues_identically(tmp_path, toy_problem):
    space, toy = toy_problem
    full = ablation.run_isolated_optimizer("PSO", space, ToyEvaluator(toy), 120, 40, 1, str(tmp_path / "a"))
    os.makedirs(tmp_path / "b")
    # run to 80 evaluations, "crash", then resume to 120
    ablation.run_isolated_optimizer("PSO", space, ToyEvaluator(toy), 80, 40, 1, str(tmp_path / "b"))
    resumed = ablation.run_isolated_optimizer("PSO", space, ToyEvaluator(toy), 120, 40, 1, str(tmp_path / "b"))
    assert [round(i.fitness, 9) for i in resumed["inds"]] == [round(i.fitness, 9) for i in full["inds"]]


# ---------------------------------------------------------------- refit / baselines / inference
def test_shared_refit_early_stops_on_train_rows_and_measures_inference(fake_covertype, monkeypatch, tmp_path):
    import evosynth.models.refit as refit
    seen = {}
    real_fit = refit.fit_model

    def spy(model, mt, X_tr, y_tr, X_es=None, y_es=None, **kw):
        seen["n_es"] = 0 if y_es is None else len(y_es)
        return real_fit(model, mt, X_tr, y_tr, X_es, y_es, **kw)
    monkeypatch.setattr(refit, "fit_model", spy)
    cfg = {"model_type": "LightGBM", "fe_strategy": "none", "n_estimators": 50, "learning_rate": 0.1,
           "max_depth": 3, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0}
    r = refit.refit_and_score(cfg, "covertype", verbose=False, model_path=tmp_path / "m.pkl")
    assert seen["n_es"] == 60                                # 15% of the 400 train rows
    assert r["test_rows"] == 200 and r["train_rows"] == 400 and 33 < r["test_score"] <= 100
    assert {"single_row_ms", "batch_rows_per_s", "model_mb"} <= set(r["inference"])
    assert pickle.load(open(tmp_path / "m.pkl", "rb"))["config"] == cfg


def test_defaults_baseline_trains_all_four_families(fake_covertype, tmp_path):
    from evosynth.experiments.baselines import run_defaults
    r = run_defaults(str(tmp_path), dataset="covertype")
    assert set(r["models"]) == {"XGBoost", "LightGBM", "RandomForest", "ExtraTrees"}
    assert r["test_score"] == max(m["test_score"] for m in r["models"].values())


def test_with_overrides_keeps_existing_overrides():
    base = load_config(overrides=["bandit.c=0.3"])
    s = with_overrides(base, ["evaluation.early_stopping=false"])
    assert s.bandit.c == 0.3 and s.evaluation.early_stopping is False and base.evaluation.early_stopping is True


# ---------------------------------------------------------------- summary
def _fake_run(root, exp, ds, variant, seed, result, status="completed"):
    d = root / exp / ds / variant / f"seed_{seed}"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"status": status}))
    (d / "result.json").write_text(json.dumps(result))
    return d


def test_summary_tables_are_generated_from_run_folders(tmp_path):
    for seed, score in ((1, 90.0), (2, 92.0)):
        d = _fake_run(tmp_path, "search", "adult", "search_full", seed,
                      {"metric": "roc_auc", "refit": {"test_score": score, "inference": {"single_row_ms": 1.0, "model_mb": 2.0}}})
        (d / "population.csv").write_text("Eval_ID,Phase,Algorithm,Fitness\n1,Init,Init,80\n2,T1,T1_GA,85\n3,T1,T1_GA,84\n")
    _fake_run(tmp_path, "search", "adult", "search_full", 3, {"metric": "roc_auc", "refit": {"test_score": 10.0}},
              status="failed")                               # failed runs are ignored
    _fake_run(tmp_path, "baselines", "adult", "defaults", 1,
              {"metric": "roc_auc", "models": {"XGBoost": {"test_score": 88.0}, "ExtraTrees": {"test_score": 86.0}}})
    for seed, delta in ((1, 0.5), (2, -0.2)):
        _fake_run(tmp_path, "ablation", "adult", "CVGS-500", seed,
                  {"metric": "roc_auc", "checkpoint_eval": 125, "delta_vs_best_isolated": delta,
                   "methods": {"EvoSynth": {"best_cv_score": 90 + delta, "score_at_mark": 85, "curve": [[50, 80], [500, 90]]},
                               "GA": {"best_cv_score": 90, "score_at_mark": 84, "curve": [[50, 79], [500, 90]]}}})
    out = build_summary(tmp_path)
    main = (out / "main_table.md").read_text(encoding="utf-8")
    assert "| adult | EvoSynth (search_full) | roc_auc | 91.00 ± 1.41 | 2 |" in main
    assert "Default XGBoost" in main and "Default ExtraTrees" in main
    abl = (out / "ablation_table.md").read_text(encoding="utf-8")
    assert "1/2" in abl                                      # EvoSynth beat the best isolated in 1 of 2 seeds
    conv = (out / "convergence" / "search_adult_search_full.csv").read_text(encoding="utf-8").splitlines()
    assert conv[1].startswith("1,80.0") and conv[3].startswith("3,85.0")   # best so far, never decreases
