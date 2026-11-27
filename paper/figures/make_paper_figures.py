"""
Figures for paper.tex, drawn from the experiment outputs:
    python paper/figures/make_paper_figures.py
Writes ablation_curves.pdf (Fig. 2): best search CV accuracy so far vs evaluations used, for
GA, PSO, DE and CMA-ES run alone and for EvoSynth (CVGS-500, Covertype, seed 42).
Distinct markers and line styles keep the series apart in grayscale print.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "results", "ablation", "covertype", "cvgs_500", "seed_42", "result.json")

STYLE = {  # name: (label, color, marker, linestyle, width)
    "GA": ("GA alone", "#f47230", "v", "--", 1.2),
    "PSO": ("PSO alone", "#00a888", "s", "--", 1.2),
    "DE": ("DE alone", "#d28e05", "^", "--", 1.2),
    "CMAES": ("CMA-ES alone", "#e24d8b", "D", "--", 1.2),
    "EvoSynth": ("EvoSynth", "#1b64be", "o", "-", 2.0),
}

d = json.load(open(SRC, encoding="utf-8"))
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(3.45, 2.25))
for name, (label, color, marker, ls, lw) in STYLE.items():
    xs, ys = zip(*d["methods"][name]["curve"])
    ax.step(xs, ys, where="post", color=color, linestyle=ls, linewidth=lw, label=label)
    ax.plot(xs, ys, linestyle="none", marker=marker, markersize=3.2, color=color)
ax.axvline(d["checkpoint_eval"], color="0.6", linewidth=0.6, linestyle=":")
ax.text(d["checkpoint_eval"] + 6, 81.97, "25%", fontsize=6.5, color="0.35")
ax.set_xlabel("Evaluations used")
ax.set_ylabel("Best search CV accuracy (%)")
ax.set_xlim(40, 515)
ax.set_ylim(81.3, 82.45)
ax.grid(axis="y", color="0.9", linewidth=0.5)
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=6.3, frameon=False, ncol=2, borderaxespad=0.1)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
out = os.path.join(HERE, "ablation_curves.pdf")
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=200)
print("wrote", out)
