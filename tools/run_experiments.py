"""
Phase 5 experiment runner: runs a named plan of `evosynth` commands one after another.

    py -3.14 tools/run_experiments.py smoke            # the plan chosen for Phase 5
    py -3.14 tools/run_experiments.py core --dry-run   # list the jobs and what is already done

Plans (all use the same --subsample rows for every method's search, a subset of each dataset's
published search rows; see settings/datasets.toml and the EvoSynth-Bench release):

  smoke  seed 42: main comparison on Covertype and Adult; ablation CVGS-500 on Covertype;
         speed-up benchmark; NAS on Fashion-MNIST
  lean   seeds 42-44: main comparison on Covertype, HIGGS, Adult, Poker Hand; ablation CVGS-500
         with the tier-split ablation on Covertype; speed-up; NAS
  core   seeds 42-44: main comparison on all 7 tabular datasets; the rest as in lean

Main comparison, per dataset and seed (equal budgets):
  EvoSynth (preset search_full, 500 evaluations) -> Random Search (500) -> Optuna TPE (500)
  -> untuned defaults -> FLAML with the same wall time as that EvoSynth search.

A job whose run folder already says status "completed" is skipped, so the runner can be stopped
(Ctrl+C) and started again; an interrupted ablation or Optuna study resumes from its checkpoint,
any other interrupted run starts over. Progress is appended to results/runner.log. When the plan
is finished, `evosynth summary` builds results/summary/.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evosynth.experiments.run_context import run_path   # noqa: E402
from evosynth.paths import RESULTS_DIR                   # noqa: E402

SUBSAMPLE = 10000          # search rows for every method (user choice for Phase 5)
BUDGET = 500               # EvoSynth search_full; Random Search / Optuna get the same number of evaluations
WORKERS = 8                # parallel evaluation threads for the ablation (settings default is 2)

PLANS = {
    "smoke": {"seeds": [42], "datasets": ["covertype", "adult"], "splits": "stratified"},
    "lean": {"seeds": [42, 43, 44], "datasets": ["covertype", "higgs", "adult", "poker_hand"],
             "splits": "stratified,random,elite"},
    "core": {"seeds": [42, 43, 44], "datasets": ["covertype", "higgs", "adult", "airlines", "poker_hand",
                                                  "miniboone", "jannis"], "splits": "stratified,random,elite"},
}


def _evosynth_seconds(dataset, seed):
    """Wall time of the EvoSynth search for (dataset, seed), for FLAML's time budget."""
    path = Path(run_path("search", dataset, "search_full", seed)) / "result.json"
    try:
        return max(60, int(json.loads(path.read_text(encoding="utf-8"))["search_seconds"]))
    except (OSError, KeyError, ValueError):
        return None


def jobs(plan):
    p = PLANS[plan]
    out = []
    for seed in p["seeds"]:
        s = ["--seed", str(seed)]
        for ds in p["datasets"]:
            sub = ["--dataset", ds, "--subsample", str(SUBSAMPLE)]
            out.append((f"EvoSynth search {ds} s{seed}", ("search", ds, "search_full", seed),
                        ["run", "--full"] + sub + s))
            out.append((f"Random Search {ds} s{seed}", ("baselines", ds, "random_search", seed),
                        ["baselines", "--test", "random", "--trials", str(BUDGET)] + sub + s))
            out.append((f"Optuna TPE {ds} s{seed}", ("baselines", ds, "optuna_tpe", seed),
                        ["baselines", "--test", "optuna", "--trials", str(BUDGET)] + sub + s))
            out.append((f"Untuned defaults {ds} s{seed}", ("baselines", ds, "defaults", seed),
                        ["baselines", "--test", "defaults", "--dataset", ds] + s))
            out.append((f"FLAML {ds} s{seed}", ("baselines", ds, "flaml", seed),
                        lambda ds=ds, seed=seed, s=s: ["baselines", "--test", "flaml", "--dataset", ds, "--time-budget",
                                                       str(_evosynth_seconds(ds, seed) or 600)] + s))
        out.append((f"Ablation CVGS-500 covertype s{seed}", ("ablation", "covertype", "CVGS-500", seed),
                    ["ablation", "--setup", "CVGS-500", "--dataset", "covertype", "--subsample", str(SUBSAMPLE),
                     "--max-workers", str(WORKERS), "--splits", p["splits"]] + s))
        out.append((f"Speed-up covertype s{seed}", ("speedup", "covertype", "default", seed),
                    ["speedup", "--dataset", "covertype", "--subsample", str(SUBSAMPLE)] + s))
        out.append((f"NAS fashion_mnist s{seed}", ("nas", "fashion_mnist", "default", seed), ["nas"] + s))
    return out


def _done(where):
    experiment, dataset, variant, seed = where
    manifest = Path(run_path(experiment, dataset, variant, seed)) / "manifest.json"
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("status") == "completed"
    except (OSError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser(description="Run a plan of EvoSynth experiments (Phase 5).")
    ap.add_argument("plan", choices=sorted(PLANS))
    ap.add_argument("--dry-run", action="store_true", help="list the jobs and their status, run nothing")
    a = ap.parse_args()

    todo = jobs(a.plan)
    log = Path(RESULTS_DIR) / "runner.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    def note(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        if not a.dry_run:
            with open(log, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    note(f"plan '{a.plan}': {len(todo)} jobs")
    for i, (label, where, args) in enumerate(todo, 1):
        if _done(where):
            note(f"[{i}/{len(todo)}] done already: {label}")
            continue
        if a.dry_run:
            note(f"[{i}/{len(todo)}] to run: {label}")
            continue
        args = args() if callable(args) else args
        note(f"[{i}/{len(todo)}] START {label}: evosynth {' '.join(args)}")
        t = time.time()
        code = subprocess.call([sys.executable, "-m", "evosynth"] + args, cwd=ROOT,
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # each run keeps its own run.log
        note(f"[{i}/{len(todo)}] {'OK  ' if code == 0 else f'FAILED (exit {code})'} {label} "
             f"in {(time.time() - t) / 60:.1f} min")
    if not a.dry_run:
        code = subprocess.call([sys.executable, "-m", "evosynth", "summary"], cwd=ROOT,
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        note(f"summary {'built' if code == 0 else f'FAILED (exit {code})'}: results/summary/")


if __name__ == "__main__":
    main()
