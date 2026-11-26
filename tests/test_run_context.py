"""Run folders: manifest.json, result.json, run.log, events.jsonl (R5)."""
import json
import sys

import numpy as np
import pytest

from evosynth.engine.controller import DynamicOptimizer
from evosynth.experiments.run_context import RunContext, run_path


def test_run_path_layout(tmp_path):
    p = run_path("search", "covertype", "Search-Demo", seed=7, results_dir=tmp_path)
    assert p == str(tmp_path / "search" / "covertype" / "search_demo" / "seed_7")


def test_completed_run_writes_all_files(tmp_path, toy_problem):
    space, evaluator = toy_problem
    stdout = sys.stdout
    np.random.seed(0)
    with RunContext("search", "toy", "smoke", results_dir=tmp_path) as run:
        opt = DynamicOptimizer(space, evaluator, budget=80, pop_size=40, t1_iters=2, k_iters=1,
                               max_workers=1, population_csv=run.file("population.csv"))
        best = opt.run()
        run.write_result({"best": best.fitness})
    assert sys.stdout is stdout

    folder = tmp_path / "search" / "toy" / "smoke" / "seed_42"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["settings"]["engine"]["n_promoted"] == 2
    assert set(manifest["git"]) == {"commit", "branch", "dirty"}
    assert manifest["environment"]["packages"]["numpy"]

    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    assert result["experiment"] == "search" and result["best"] == pytest.approx(best.fitness)

    assert "FINAL RESULTS" in (folder / "run.log").read_text(encoding="utf-8")
    events = [json.loads(l) for l in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {"step", "progress"} <= {e["type"] for e in events}
    assert all(e["type"] != "log" for e in events)
    assert len((folder / "population.csv").read_text(encoding="utf-8").splitlines()) == 81


def test_failed_run_is_recorded(tmp_path):
    stdout = sys.stdout
    with pytest.raises(RuntimeError):
        with RunContext("search", "toy", "boom", results_dir=tmp_path):
            print("about to fail")
            raise RuntimeError("boom")
    assert sys.stdout is stdout
    manifest = json.loads((tmp_path / "search/toy/boom/seed_42/manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "RuntimeError: boom" in manifest["error"]
    assert not (tmp_path / "search/toy/boom/seed_42/result.json").exists()
