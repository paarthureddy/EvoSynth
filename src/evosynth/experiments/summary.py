"""
Summary tables generated from the run folders (Phase 4.7). Command: `evosynth summary`.

Reads every completed run (manifest.json status "completed") under
results/<experiment>/<dataset>/<variant>/seed_<k>/ and writes results/summary/:

  main_table.md / .csv     held-out test score per dataset and method: mean ± std over seeds, n,
                           plus measured inference cost (search, scale, baselines)
  ablation_table.md / .csv search CV score at the comparison mark and at the full budget per
                           dataset, setup and method (mean ± std), and how often EvoSynth beat the
                           best isolated optimizer
  nas_table.md / .csv      NAS validation accuracy
  convergence/<experiment>_<dataset>_<variant>.csv   best-so-far score after each evaluation,
                           mean and std over seeds (from population.csv), and the ablation curves
                           at their scoring marks

Nothing is typed in by hand: re-running this after new runs regenerates every table.
"""
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from evosynth.paths import RESULTS_DIR


def _runs(root, experiment):
    """Yield (dataset, variant, seed, result, run_dir) for completed runs of one experiment."""
    base = Path(root) / experiment
    if not base.exists():
        return
    for manifest in sorted(base.glob("*/*/seed_*/manifest.json")):
        run_dir = manifest.parent
        try:
            status = json.loads(manifest.read_text(encoding="utf-8")).get("status")
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if status != "completed":
            continue
        dataset, variant = run_dir.parent.parent.name, run_dir.parent.name
        yield dataset, variant, int(run_dir.name.split("_", 1)[1]), result, run_dir


def _stat(values):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return None, None, 0
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0), len(v)


def _pm(mean, std, n, digits=2):
    if mean is None:
        return "-"
    return f"{mean:.{digits}f} ± {std:.{digits}f}" if n > 1 else f"{mean:.{digits}f}"


def _write(path_md, path_csv, title, header, rows, note=""):
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([h[0] for h in header])
        for r in rows:
            w.writerow([r.get(h[0], "") for h in header])
    with open(path_md, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{note}\n\n" if note else f"# {title}\n\n")
        f.write("| " + " | ".join(h[1] for h in header) + " |\n")
        f.write("|" + "|".join(" --- " for _ in header) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(h[2] if len(h) > 2 else h[0], "")) for h in header) + " |\n")


def _main_table(root):
    groups = defaultdict(lambda: defaultdict(list))   # (dataset, method) -> field -> values

    def add(dataset, method, seed, metric, score, inference=None):
        g = groups[(dataset, method)]
        g["seeds"].append(seed); g["score"].append(score); g["metric"].append(metric)
        g["ms"].append((inference or {}).get("single_row_ms")); g["mb"].append((inference or {}).get("model_mb"))

    for d, v, seed, r, _ in _runs(root, "search"):
        if r.get("refit") and "test_score" in r["refit"]:
            add(d, f"EvoSynth ({v})", seed, r["metric"], r["refit"]["test_score"], r["refit"].get("inference"))
    for d, v, seed, r, _ in _runs(root, "scaling"):
        if "test_score" in r:
            add(d, f"EvoSynth scale ({v})", seed, r["metric"], r["test_score"], r.get("inference"))
    for d, v, seed, r, _ in _runs(root, "baselines"):
        if v == "defaults":
            for m, mr in r.get("models", {}).items():
                add(d, f"Default {m}", seed, r["metric"], mr["test_score"], mr.get("inference"))
        elif "test_score" in r:
            inf = r.get("inference") if isinstance(r.get("inference"), dict) and "error" not in r.get("inference", {}) else None
            add(d, {"random_search": "Random Search", "optuna_tpe": "Optuna TPE", "flaml": "FLAML"}.get(v, v),
                seed, r["metric"], r["test_score"], inf)

    rows = []
    for (d, method), g in sorted(groups.items()):
        m, s, n = _stat(g["score"])
        ms, _, _ = _stat(g["ms"]); mb, _, _ = _stat(g["mb"])
        rows.append({"dataset": d, "method": method, "metric": g["metric"][0], "test_mean": round(m, 4),
                     "test_std": round(s, 4), "n_seeds": n, "seeds": " ".join(map(str, sorted(g["seeds"]))),
                     "test": _pm(m, s, n), "single_row_ms": "-" if ms is None else round(ms, 3),
                     "model_mb": "-" if mb is None else round(mb, 2)})
    return rows


def _ablation_table(root):
    groups = defaultdict(lambda: defaultdict(list))
    wins = defaultdict(lambda: [0, 0])
    curves = defaultdict(list)
    for d, setup, seed, r, _ in _runs(root, "ablation"):
        for method, m in r.get("methods", {}).items():
            g = groups[(d, setup, method)]
            g["final"].append(m["best_cv_score"]); g["mark"].append(m["score_at_mark"])
            g["metric"].append(r.get("metric")); g["mark_evals"].append(r.get("checkpoint_eval"))
            curves[(d, setup, method)].append(dict(m.get("curve", [])))
        if "delta_vs_best_isolated" in r:
            wins[(d, setup)][0] += r["delta_vs_best_isolated"] > 0
            wins[(d, setup)][1] += 1
    rows = []
    for (d, setup, method), g in sorted(groups.items()):
        fm, fs, n = _stat(g["final"]); mm, ms, _ = _stat(g["mark"])
        w = wins[(d, setup)]
        rows.append({"dataset": d, "setup": setup, "method": method, "metric": g["metric"][0],
                     "mark_evals": g["mark_evals"][0], "at_mark": _pm(mm, ms, n), "at_budget": _pm(fm, fs, n),
                     "budget_mean": round(fm, 4), "budget_std": round(fs, 4), "n_seeds": n,
                     "evosynth_beat_best_isolated": f"{w[0]}/{w[1]}" if method == "EvoSynth" else ""})
    return rows, curves


def _convergence(root, out_dir):
    written = []
    for experiment in ("search", "scaling"):
        by = defaultdict(list)
        for d, v, seed, r, run_dir in _runs(root, experiment):
            pop = run_dir / "population.csv"
            if pop.exists():
                with open(pop, encoding="utf-8") as f:
                    fit = [float(row["Fitness"]) for row in csv.DictReader(f)]
                by[(d, v)].append(np.maximum.accumulate(fit))
        for (d, v), curves in by.items():
            n = min(len(c) for c in curves)
            arr = np.array([c[:n] for c in curves])
            path = out_dir / f"{experiment}_{d}_{v}.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["eval", "best_mean", "best_std", "n_seeds"])
                for i in range(n):
                    w.writerow([i + 1, round(arr[:, i].mean(), 4),
                                round(arr[:, i].std(ddof=1), 4) if len(arr) > 1 else 0.0, len(arr)])
            written.append(path)
    return written


def build_summary(results_dir=None):
    root = Path(results_dir or RESULTS_DIR)
    out = root / "summary"
    (out / "convergence").mkdir(parents=True, exist_ok=True)

    main = _main_table(root)
    _write(out / "main_table.md", out / "main_table.csv", "Held-out test score",
           [("dataset", "Dataset"), ("method", "Method"), ("metric", "Metric"),
            ("test_mean", "Test (mean ± std over seeds)", "test"), ("n_seeds", "Seeds"),
            ("single_row_ms", "Latency ms / row"), ("model_mb", "Model MB"),
            ("test_std", "Std", "test_std"), ("seeds", "Seed list")], main,
           "Generated by `evosynth summary` from results/*/<dataset>/<variant>/seed_*/result.json. "
           "Test rows are never seen by any search.")

    abl, curves = _ablation_table(root)
    _write(out / "ablation_table.md", out / "ablation_table.csv", "Ablation: search CV score",
           [("dataset", "Dataset"), ("setup", "Setup"), ("method", "Method"), ("metric", "Metric"),
            ("at_mark", "At the mark"), ("mark_evals", "Mark (evals)"),
            ("budget_mean", "At full budget (mean ± std)", "at_budget"), ("n_seeds", "Seeds"),
            ("evosynth_beat_best_isolated", "EvoSynth > best isolated (seeds)"),
            ("budget_std", "Std")], abl,
           "Generated by `evosynth summary` from results/ablation/*/<setup>/seed_*/result.json.")
    for (d, setup, method), runs in curves.items():
        marks = sorted(set.intersection(*(set(c) for c in runs))) if runs else []
        with open(out / "convergence" / f"ablation_{d}_{setup}_{method.replace('[', '_').replace(']', '')}.csv",
                  "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["eval", "best_mean", "best_std", "n_seeds"])
            for m in marks:
                vals = [c[m] for c in runs]
                w.writerow([m, round(float(np.mean(vals)), 4),
                            round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0, len(vals)])

    nas = defaultdict(list)
    for d, v, seed, r, _ in _runs(root, "nas"):
        nas[(d, v)].append(r.get("best_score"))
    nas_rows = [{"dataset": d, "variant": v, "validation": _pm(*_stat(vals)), "n_seeds": len(vals)}
                for (d, v), vals in sorted(nas.items())]
    _write(out / "nas_table.md", out / "nas_table.csv", "NAS: validation accuracy",
           [("dataset", "Dataset"), ("variant", "Variant"), ("validation", "Validation accuracy"),
            ("n_seeds", "Seeds")], nas_rows)

    conv = _convergence(root, out / "convergence")
    print(f"  [SUMMARY] {len(main)} method rows, {len(abl)} ablation rows, {len(nas_rows)} NAS rows, "
          f"{len(conv)} convergence curves -> {out}")
    return out
