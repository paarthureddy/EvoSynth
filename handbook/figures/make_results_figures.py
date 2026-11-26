"""
Regenerate the result figures of chapter 08 from the experiment outputs:
    python handbook/figures/make_results_figures.py
Reads results/summary/*.csv (built by `evosynth summary`) and the speed-up result.json.
Every number drawn here is also printed in a table in 08-results.md.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SUMMARY = os.path.join(ROOT, "results", "summary")

INK, MUTED, GRID, BG = "#1f2a24", "#5b675f", "#e3e8e5", "#ffffff"
# categorical slots in fixed order (validated: scripts/validate_palette.js, light surface)
SERIES = {"EvoSynth": "#257de0", "GA": "#f47230", "PSO": "#00b894", "DE": "#f2a600", "CMAES": "#ee7fac"}
HIGHLIGHT, OTHER, FLOOR = "#257de0", "#7d8a83", "#b9c3bd"
FONT = "font-family='Segoe UI, Helvetica, Arial, sans-serif'"


def svg(name, w, h, body, title):
    s = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='{w}' height='{h}' role='img' "
         f"aria-label='{title}' {FONT}><title>{title}</title>"
         f"<rect width='{w}' height='{h}' rx='14' fill='{BG}'/>{body}</svg>\n")
    open(os.path.join(HERE, name), "w", encoding="utf-8").write(s)
    print("wrote", name)


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{fill}' text-anchor='{anchor}' "
            f"font-weight='{weight}'>{s}</text>")


def ln(x1, y1, x2, y2, stroke=GRID, w=1):
    return f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{stroke}' stroke-width='{w}'/>"


def read_csv(name):
    with open(os.path.join(SUMMARY, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def short(method):
    return method.replace(" (search_full)", "").replace("Default ", "Untuned ")


# ------------------------------------------------------------------ test scores (dot plot)
def test_scores():
    rows = read_csv("main_table.csv")
    panels = [("covertype", "Covertype · test accuracy (%)", 84.0, 94.5, 1.0),
              ("adult", "Adult · test ROC AUC (%)", 88.5, 93.5, 1.0)]
    W, row_h, top = 760, 26, 62
    n = max(sum(r["dataset"] == d for r in rows) for d, *_ in panels)
    ph = n * row_h + 60
    H = top + len(panels) * ph + 30
    b = t(30, 32, "Held-out test score of every method (seed 42, 500 evaluations for the tuned searches)", 15, weight="bold")
    x0, x1 = 210, W - 70
    for p, (ds, title, lo, hi, step) in enumerate(panels):
        y0 = top + p * ph
        sel = sorted((r for r in rows if r["dataset"] == ds), key=lambda r: float(r["test_mean"]), reverse=True)
        b += t(30, y0 + 4, title, 13, weight="bold")
        X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
        v = lo
        while v <= hi + 1e-9:
            b += ln(X(v), y0 + 14, X(v), y0 + 14 + n * row_h) + t(X(v), y0 + 30 + n * row_h, f"{v:g}", 11, MUTED, "middle")
            v += step
        for i, r in enumerate(sel):
            y = y0 + 14 + i * row_h + row_h / 2
            m, val = short(r["method"]), float(r["test_mean"])
            col = HIGHLIGHT if m == "EvoSynth" else (FLOOR if m.startswith("Untuned") else OTHER)
            b += t(x0 - 12, y + 4, m, 12, INK, "end", "bold" if m == "EvoSynth" else "normal")
            b += ln(x0, y, X(val), y, GRID, 2)
            b += f"<circle cx='{X(val):.1f}' cy='{y:.1f}' r='6' fill='{col}' stroke='{BG}' stroke-width='2'/>"
            b += t(X(val) + 10, y + 4, f"{val:.2f}", 11, INK)
    b += t(30, H - 14, "Axes start above zero to show small differences: a 0.1-point gap on Covertype is about 20 test rows.", 11, MUTED)
    svg("results_test.svg", W, H, b, "Held-out test score of every method on Covertype and Adult")


# ------------------------------------------------------------------ search time (bars from zero)
def search_time():
    data = []
    for ds in ("covertype", "adult"):
        for name, path in [("EvoSynth", f"search/{ds}/search_full"), ("Random Search", f"baselines/{ds}/random_search"),
                           ("Optuna TPE", f"baselines/{ds}/optuna_tpe")]:
            d = json.load(open(os.path.join(ROOT, "results", path, "seed_42", "result.json"), encoding="utf-8"))
            data.append((ds, name, d["search_seconds"] / 60))
    W, H = 760, 285
    b = t(30, 32, "Search wall time for 500 evaluations (minutes; EvoSynth evaluates in parallel)", 15, weight="bold")
    x0, x1, mx = 200, W - 90, 55
    X = lambda v: x0 + v / mx * (x1 - x0)
    for v in range(0, 56, 10):
        b += ln(X(v), 56, X(v), 262) + t(X(v), 278, str(v), 11, MUTED, "middle")
    y = 60
    for ds in ("covertype", "adult"):
        b += t(30, y + 14, ds.capitalize(), 13, weight="bold")
        for _, name, mins in [d for d in data if d[0] == ds]:
            col = HIGHLIGHT if name == "EvoSynth" else OTHER
            b += t(x0 - 12, y + 14, name, 12, INK, "end", "bold" if name == "EvoSynth" else "normal")
            b += f"<rect x='{x0:.1f}' y='{y + 2:.1f}' width='{max(X(mins) - x0, 3):.1f}' height='16' rx='4' fill='{col}'/>"
            b += t(X(mins) + 8, y + 15, f"{mins:.1f} min", 11, INK)
            y += 24
        y += 22
    svg("results_time.svg", W, H, b, "Search wall time of EvoSynth, Random Search and Optuna")


# ------------------------------------------------------------------ ablation curves (lines)
def ablation():
    d = json.load(open(os.path.join(ROOT, "results", "ablation", "covertype", "cvgs_500", "seed_42", "result.json"),
                       encoding="utf-8"))
    W, H = 760, 380
    x0, x1, y0, y1 = 80, W - 150, 60, 310
    lo, hi = 81.2, 82.5
    X = lambda e: x0 + e / 500 * (x1 - x0)
    Y = lambda v: y1 - (v - lo) / (hi - lo) * (y1 - y0)
    b = t(30, 32, "Ablation CVGS-500 (Covertype): best search CV accuracy so far, each method alone vs EvoSynth", 15, weight="bold")
    for v in (81.2, 81.4, 81.6, 81.8, 82.0, 82.2, 82.4):
        b += ln(x0, Y(v), x1, Y(v)) + t(x0 - 8, Y(v) + 4, f"{v:.1f}", 11, MUTED, "end")
    for e in range(0, 501, 100):
        b += t(X(e), y1 + 18, str(e), 11, MUTED, "middle")
    b += t((x0 + x1) / 2, y1 + 38, "evaluations used", 12, MUTED, "middle")
    b += ln(X(125), y0, X(125), y1, "#9aa59f", 1) + t(X(125) + 4, y0 + 10, "25% mark (125)", 11, MUTED)
    labels = []
    for name in ("GA", "PSO", "DE", "CMAES", "EvoSynth"):
        pts = d["methods"][name]["curve"]
        path = ""
        for i, (e, v) in enumerate(pts):   # step line: best-so-far only changes at a measurement
            if i == 0:
                path = f"M{X(e):.1f},{Y(v):.1f}"
            else:
                path += f" H{X(e):.1f} V{Y(v):.1f}"
        w = 3 if name == "EvoSynth" else 2
        b += f"<path d='{path}' fill='none' stroke='{SERIES[name]}' stroke-width='{w}' stroke-linejoin='round'/>"
        labels.append([Y(pts[-1][1]), name, pts[-1][1]])
    labels.sort()
    for i in range(1, len(labels)):            # keep end labels 14 px apart
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + 14)
    for yy, name, v in labels:
        b += f"<circle cx='{x1 + 10:.1f}' cy='{yy - 4:.1f}' r='4' fill='{SERIES[name]}'/>"
        b += t(x1 + 18, yy, f"{'CMA-ES' if name == 'CMAES' else name} {v:.2f}", 11, INK, weight="bold" if name == "EvoSynth" else "normal")
    b += t(30, H - 12, "Measured at 50, 100, 125, 150, …, 500 evaluations. EvoSynth, PSO and CMA-ES reach the same 82.35% model (see §3).", 11, MUTED)
    svg("results_ablation.svg", W, H, b, "Ablation convergence curves for GA, PSO, DE, CMA-ES and EvoSynth")


# ------------------------------------------------------------------ speed-up (bars from zero)
def speedup():
    d = json.load(open(os.path.join(ROOT, "results", "speedup", "covertype", "default", "seed_42", "result.json"),
                       encoding="utf-8"))
    conds = list(d["conditions"].items())
    W, H = 760, 250
    x0, x1, mx = 260, W - 120, 270
    X = lambda v: x0 + v / mx * (x1 - x0)
    b = t(30, 32, f"Time to evaluate {d['configurations']} configurations on Covertype ({d['cpu_count']} CPU threads)", 15, weight="bold")
    for v in range(0, 271, 50):
        b += ln(X(v), 52, X(v), 50 + len(conds) * 34) + t(X(v), 66 + len(conds) * 34, f"{v} s", 11, MUTED, "middle")
    for i, (name, c) in enumerate(conds):
        y = 54 + i * 34
        col = HIGHLIGHT if c["workers"] > 1 and c["early_stopping"] else OTHER
        b += t(x0 - 12, y + 15, name, 12, INK, "end")
        b += f"<rect x='{x0:.1f}' y='{y + 2:.1f}' width='{X(c['seconds']) - x0:.1f}' height='18' rx='4' fill='{col}'/>"
        b += t(X(c["seconds"]) + 8, y + 16, f"{c['seconds']:.0f} s · {c['speedup']:.2f}×", 11, INK, weight="bold" if col == HIGHLIGHT else "normal")
    b += t(30, H - 14, "Speed-up = sequential-without-early-stopping time ÷ condition time. Same 40 configurations in every condition.", 11, MUTED)
    svg("results_speedup.svg", W, H, b, "Evaluation speed-up from parallel threads and early stopping")


if __name__ == "__main__":
    test_scores()
    search_time()
    ablation()
    speedup()
