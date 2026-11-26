"""
Regenerate the handbook's SVG figures:  python handbook/figures/make_figures.py
Conceptual figures are drawn by hand; data figures use the numbers quoted in the chapters
(worked examples) or a fixed random seed (illustrations), so they are reproducible.
"""
import math
import os
import random

OUT = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, LINE, BG = "#1f2a24", "#5b675f", "#cfd8d2", "#ffffff"
GREEN, GREEN_SOFT, RED, RED_SOFT = "#2f6b4f", "#e2efe7", "#c0503f", "#fbe4e0"
BLUE, BLUE_SOFT, PURPLE, PURPLE_SOFT = "#3d73c4", "#e1ebf8", "#7b5cc4", "#eee8f8"
AMBER, AMBER_SOFT, GREY_SOFT = "#a76d00", "#f7ecd5", "#eef1ef"
FONT = "font-family='Segoe UI, Helvetica, Arial, sans-serif'"


def svg(name, w, h, body, title):
    s = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='{w}' height='{h}' role='img' "
         f"aria-label='{title}' {FONT}><title>{title}</title>"
         f"<rect width='{w}' height='{h}' rx='14' fill='{BG}'/>{body}</svg>\n")
    open(os.path.join(OUT, name), "w", encoding="utf-8").write(s)


def t(x, y, s, size=13, fill=INK, anchor="start", weight="normal"):
    return f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{fill}' text-anchor='{anchor}' font-weight='{weight}'>{s}</text>"


def r(x, y, w, h, fill, stroke="none", rx=6):
    return f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' fill='{fill}' stroke='{stroke}'/>"


def line(x1, y1, x2, y2, stroke=INK, w=1.5, dash="", arrow=False):
    d = f" stroke-dasharray='{dash}'" if dash else ""
    m = " marker-end='url(#a)'" if arrow else ""
    return f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{stroke}' stroke-width='{w}'{d}{m}/>"


def arrowdef(color=INK):
    return (f"<defs><marker id='a' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto'>"
            f"<path d='M0,0 L10,5 L0,10 z' fill='{color}'/></marker></defs>")


def circle(x, y, rad, fill, stroke="none"):
    return f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{rad}' fill='{fill}' stroke='{stroke}'/>"


# ------------------------------------------------------------------ 00a
def split():
    b = t(30, 34, "How one dataset is divided (Covertype, 581,012 rows; data_seed fixes it)", 15, weight="bold")
    x0, W = 30, 660
    parts = [("test 20,000", 20000, "#9aa59f", "split off first · never searched · scored once"),
             ("search 20,000", 20000, BLUE, "cross-validation during the search"),
             ("train 60,000", 60000, AMBER, "refit of the selected configuration")]
    total = 581012
    x = x0
    for i, (lab, n, col, note) in enumerate(parts):
        w = max(W * n / total * 3.2, 125)
        b += r(x, 60, w, 34, col) + t(x + 6, 82, lab, 12, "#ffffff", weight="bold")
        b += t(30, 130 + i * 24, "■", 14, col) + t(48, 130 + i * 24, f"<tspan font-weight='bold'>{lab}</tspan>: {note}", 13)
        x += w + 4
    b += r(x, 60, x0 + W - x, 34, GREY_SOFT) + t(x + 8, 82, "rest (unused)", 12, MUTED)
    b += t(30, 210, "search and train rows are drawn from what is left after the test rows, so neither ever overlaps test.", 12, MUTED)
    b += t(30, 228, "(widths not to scale)", 11, MUTED)
    svg("split.svg", 720, 245, b, "Dataset split into test, search and train rows")


def cv():
    b = t(30, 32, "2-fold stratified cross-validation of one configuration", 15, weight="bold")
    for i, (tr, va) in enumerate([(0, 1), (1, 0)]):
        y = 60 + i * 70
        b += t(30, y + 22, f"round {i + 1}", 13, MUTED)
        for k in range(2):
            col, lab = (GREEN, "train") if k == tr else (BLUE, "score")
            b += r(110 + k * 230, y, 220, 34, col) + t(220 + k * 230, y + 22, f"fold {k + 1}: {lab}", 12, "#fff", "middle", "bold")
        b += t(580, y + 22, f"score{i + 1}", 13)
    b += t(30, 210, "CV score = average of score1 and score2 (each fold keeps the class proportions)", 13)
    b += t(30, 236, "Boosted models hold back 15% of the training fold for early stopping; the scored fold is never used for it.", 12, MUTED)
    svg("cv.svg", 720, 255, b, "Two-fold cross-validation")


def tree():
    b = arrowdef(MUTED) + t(30, 30, "A decision tree asks yes/no questions until it reaches a class", 15, weight="bold")
    nodes = {"root": (360, 70, "elevation > 2,800 m?"), "l": (190, 160, "distance to water &lt; 100 m?"), "r": (530, 160, "slope > 20°?"),
             "ll": (100, 250, "Aspen"), "lr": (280, 250, "Lodgepole Pine"), "rl": (450, 250, "Spruce/Fir"), "rr": (610, 250, "Krummholz")}
    for a, c, lab in [("root", "l", "no"), ("root", "r", "yes"), ("l", "ll", "yes"), ("l", "lr", "no"), ("r", "rl", "no"), ("r", "rr", "yes")]:
        (x1, y1, _), (x2, y2, _) = nodes[a], nodes[c]
        b += line(x1, y1 + 16, x2, y2 - 18, MUTED, 1.5, arrow=True) + t((x1 + x2) / 2 + 6, (y1 + y2) / 2 + 4, lab, 11, MUTED)
    for k, (x, y, lab) in nodes.items():
        leaf = len(k) == 2
        w = 8 * len(lab.replace("&lt;", "<")) + 24
        b += r(x - w / 2, y - 16, w, 32, GREEN_SOFT if leaf else BLUE_SOFT, GREEN if leaf else BLUE, 8)
        b += t(x, y + 5, lab, 12, INK, "middle", "bold" if leaf else "normal")
    b += t(30, 300, "max_depth limits how many questions a path may ask; deeper trees fit more detail but can memorise noise.", 12, MUTED)
    svg("tree.svg", 720, 318, b, "Decision tree example")


def bagging_boosting():
    b = arrowdef(MUTED) + t(30, 30, "Two ways to build a tree ensemble", 15, weight="bold")
    b += t(30, 62, "Bagging (Random Forest, ExtraTrees): independent trees on random samples, then a vote", 13, GREEN, weight="bold")
    for i in range(4):
        x = 60 + i * 120
        b += r(x, 78, 90, 44, GREEN_SOFT, GREEN) + t(x + 45, 105, f"tree {i + 1}", 12, INK, "middle")
        b += line(x + 45, 122, 600, 150, MUTED, 1.2, arrow=True)
    b += r(560, 138, 120, 34, GREEN, rx=8) + t(620, 160, "vote", 13, "#fff", "middle", "bold")
    b += t(30, 210, "Boosting (XGBoost, LightGBM): each tree fixes the errors left by the trees before it", 13, BLUE, weight="bold")
    for i in range(4):
        x = 60 + i * 150
        b += r(x, 226, 100, 44, BLUE_SOFT, BLUE) + t(x + 50, 253, f"tree {i + 1}", 12, INK, "middle")
        if i < 3:
            b += line(x + 100, 248, x + 148, 248, MUTED, 1.5, arrow=True) + t(x + 124, 240, "errors", 10, MUTED, "middle")
    b += t(30, 300, "learning_rate scales each correction; early stopping ends the sequence when a held-back slice stops improving.", 12, MUTED)
    svg("bagging_boosting.svg", 720, 318, b, "Bagging versus boosting")


def overfitting():
    W, H, x0, y0 = 600, 200, 70, 50
    b = t(30, 30, "Overfitting: training score keeps rising, validation score peaks then falls", 15, weight="bold")
    b += line(x0, y0 + H, x0 + W, y0 + H, MUTED) + line(x0, y0, x0, y0 + H, MUTED)
    tr = [(i / 60, 0.55 + 0.43 * (1 - math.exp(-4 * i / 60))) for i in range(61)]
    va = [(i / 60, 0.55 + 0.42 * (1 - math.exp(-5 * i / 60)) - 0.30 * (i / 60) ** 2) for i in range(61)]

    def path(pts):
        return " ".join(("M" if i == 0 else "L") + f"{x0 + p[0] * W:.1f},{y0 + H - (p[1] - 0.5) / 0.5 * H:.1f}" for i, p in enumerate(pts))
    b += f"<path d='{path(tr)}' fill='none' stroke='{GREEN}' stroke-width='3'/><path d='{path(va)}' fill='none' stroke='{BLUE}' stroke-width='3'/>"
    best = max(va, key=lambda p: p[1])
    bx = x0 + best[0] * W
    b += line(bx, y0, bx, y0 + H, AMBER, 1.5, "5 4") + t(bx + 6, y0 + 14, "best generalisation", 12, AMBER)
    b += t(x0 + W - 4, y0 + 18, "training score", 12, GREEN, "end", "bold") + t(x0 + W - 4, y0 + H - 50, "validation score", 12, BLUE, "end", "bold")
    b += t(x0 + W / 2, y0 + H + 26, "model complexity (more / deeper trees, weaker regularisation) →", 12, MUTED, "middle")
    b += t(30, y0 + H / 2, "score", 12, MUTED)
    svg("overfitting.svg", 720, 300, b, "Overfitting curve (illustration)")


# ------------------------------------------------------------------ 00b
def grid_random():
    b = t(30, 30, "9 evaluations: grid search vs random search", 15, weight="bold")
    rnd = random.Random(3)
    for p, (name, pts) in enumerate([("grid", [(0.17 + 0.33 * i, 0.17 + 0.33 * j) for i in range(3) for j in range(3)]),
                                     ("random", [(rnd.random(), rnd.random()) for _ in range(9)])]):
        x0, y0, S = 50 + p * 350, 60, 240
        b += r(x0, y0, S, S, GREY_SOFT, LINE, 4) + t(x0 + S / 2, y0 - 10, name, 13, INK, "middle", "bold")
        for (u, v) in pts:
            b += circle(x0 + u * S, y0 + (1 - v) * S, 6, GREEN)
            b += line(x0 + u * S, y0 + S, x0 + u * S, y0 + S + 12, BLUE, 2)
        b += t(x0 + S / 2, y0 + S + 30, "important hyperparameter →", 11, MUTED, "middle")
        distinct = len({round(u, 3) for u, _ in pts})
        b += t(x0 + S / 2, y0 + S + 48, f"{distinct} distinct values tried on it", 12, BLUE, "middle", "bold")
    b += t(30, 380, "When only a few hyperparameters matter, random search tries many more values of each.", 12, MUTED)
    svg("grid_random.svg", 720, 395, b, "Grid search versus random search")


def overtuning():
    rnd = random.Random(11)
    true = [80 + rnd.random() * 6 for _ in range(30)]
    meas = [v + rnd.gauss(0, 1.1) for v in true]
    x0, y0, W, H = 70, 50, 600, 220
    lo, hi = 76, 90

    def Y(v):
        return y0 + H - (v - lo) / (hi - lo) * H
    b = t(30, 30, "Overtuning: the highest measured score is not the best configuration", 15, weight="bold")
    b += line(x0, y0 + H, x0 + W, y0 + H, MUTED) + line(x0, y0, x0, y0 + H, MUTED)
    for v in (78, 82, 86, 90):
        b += t(x0 - 8, Y(v) + 4, f"{v}%", 11, MUTED, "end") + line(x0, Y(v), x0 + W, Y(v), LINE, 1)
    im, it = max(range(30), key=lambda i: meas[i]), max(range(30), key=lambda i: true[i])
    for i in range(30):
        x = x0 + 12 + i * (W - 24) / 29
        b += line(x, Y(true[i]), x, Y(meas[i]), LINE, 1.5)
        b += circle(x, Y(true[i]), 4, BLUE) + circle(x, Y(meas[i]), 4, AMBER if i == im else GREEN)
    xm = x0 + 12 + im * (W - 24) / 29
    xt = x0 + 12 + it * (W - 24) / 29
    b += t(xm, Y(meas[im]) - 10, "picked (luckiest)", 11, AMBER, "middle", "bold")
    b += t(xt, Y(true[it]) + 18, "truly best", 11, BLUE, "middle", "bold")
    b += circle(90, 300, 4, GREEN) + t(100, 304, "measured (noisy 2-fold CV)", 12) + circle(300, 300, 4, BLUE) + t(310, 304, "true quality", 12)
    b += t(30, 326, "Re-scoring the top few with 5-fold CV (confirm before select) reduces the noise before choosing.", 12, MUTED)
    svg("overtuning.svg", 720, 342, b, "Overtuning illustration")


# ------------------------------------------------------------------ 00c
def unit_square(x0, y0, S, label=True):
    s = r(x0, y0, S, S, GREY_SOFT, LINE, 4)
    if label:
        s += t(x0, y0 + S + 18, "0", 11, MUTED) + t(x0 + S, y0 + S + 18, "1", 11, MUTED, "end")
    return s


def P(x0, y0, S, u, v):
    return x0 + u * S, y0 + (1 - v) * S


def ga():
    x0, y0, S = 60, 50, 260
    b = arrowdef(MUTED) + t(30, 30, "GA arithmetic crossover (worked example from 00c §2)", 15, weight="bold") + unit_square(x0, y0, S)
    p1, p2, ch, mu = (0.2, 0.8), (0.6, 0.4), (0.5, 0.6), (0.5, 0.67)
    b += f"<rect x='{P(x0, y0, S, .2, .8)[0]:.1f}' y='{P(x0, y0, S, .6, .8)[1]:.1f}' width='{.4 * S:.1f}' height='{.4 * S:.1f}' fill='{GREEN_SOFT}' stroke='{GREEN}' stroke-dasharray='4 3'/>"
    for pt, lab, col in [(p1, "parent 1 (0.20, 0.80)", BLUE), (p2, "parent 2 (0.60, 0.40)", BLUE), (ch, "child (0.50, 0.60)", GREEN), (mu, "after mutation (0.50, 0.67)", AMBER)]:
        x, y = P(x0, y0, S, *pt)
        b += circle(x, y, 7, col) + t(x + 10, y - 8, lab, 11, col, weight="bold")
    b += t(360, 90, "Each child gene lies between the parents' genes:", 13)
    b += t(360, 114, "child = α·p1 + (1 − α)·p2, a fresh α per gene", 13, weight="bold")
    b += t(360, 140, "α = (0.25, 0.50) → child (0.50, 0.60)", 12, MUTED)
    b += t(360, 176, "Mutation (probability 0.1 per gene) adds", 13) + t(360, 196, "Gaussian noise σ = 0.1: here +0.07 on gene 2.", 13)
    b += t(360, 232, "The dashed box is everywhere a blend of these", 12, MUTED) + t(360, 250, "two parents can land.", 12, MUTED)
    svg("ga.svg", 720, 340, b, "GA crossover example")


def pso():
    x0, y0, S = 60, 50, 260
    b = arrowdef(INK) + t(30, 30, "PSO velocity update (1-D worked example from 00c §3, drawn on a line)", 15, weight="bold")
    X = lambda u: 60 + u * 600
    b += line(X(0), 120, X(1), 120, LINE, 3)
    for u in (0, 0.25, 0.5, 0.75, 1):
        b += line(X(u), 114, X(u), 126, MUTED) + t(X(u), 144, f"{u:.2f}", 11, MUTED, "middle")
    for u, lab, col in [(0.40, "x = 0.40", INK), (0.50, "pbest 0.50", BLUE), (0.80, "gbest 0.80", GREEN), (0.765, "new x = 0.765", AMBER)]:
        b += circle(X(u), 120, 7, col) + t(X(u), 100 if lab in ("x = 0.40", "gbest 0.80") else 172, lab, 12, col, "middle", "bold")
    comps = [("inertia 0.5·0.10", 0.05, MUTED), ("cognitive 1.5·0.5·(0.50 − 0.40)", 0.075, BLUE), ("social 1.5·0.4·(0.80 − 0.40)", 0.24, GREEN)]
    u = 0.40
    for i, (lab, dv, col) in enumerate(comps):
        y = 200 + i * 30
        b += line(X(u), y, X(u + dv), y, col, 3, arrow=True) + t(X(u + dv) + 8, y + 4, f"{lab} = +{dv:.3f}", 12, col)
        u += dv
    b += t(60, 310, "v = 0.05 + 0.075 + 0.24 = 0.365  →  x moves from 0.40 to 0.765, pulled mostly towards the swarm's best.", 12, INK)
    svg("pso.svg", 720, 330, b, "PSO velocity example")


def de():
    x0, y0, S = 60, 50, 260
    b = arrowdef(INK) + t(30, 30, "DE donor vector and trial (worked example from 00c §4)", 15, weight="bold") + unit_square(x0, y0, S)
    pts = {"a": (.3, .3), "b": (.7, .5), "c": (.5, .6), "donor": (.46, .22), "target": (.4, .9), "trial": (.46, .9)}
    cols = {"a": BLUE, "b": BLUE, "c": BLUE, "donor": GREEN, "target": PURPLE, "trial": AMBER}
    xb, yb = P(x0, y0, S, *pts["b"]); xc, yc = P(x0, y0, S, *pts["c"])
    b += line(xc, yc, xb, yb, MUTED, 2, "4 3", arrow=True) + t((xb + xc) / 2 + 6, (yb + yc) / 2 - 6, "b − c", 11, MUTED)
    xa, ya = P(x0, y0, S, *pts["a"]); xd, yd = P(x0, y0, S, *pts["donor"])
    b += line(xa, ya, xd, yd, GREEN, 2.5, arrow=True)
    for k, (u, v) in pts.items():
        x, y = P(x0, y0, S, u, v)
        dy = {"target": -10, "trial": 22}.get(k, 4)
        b += circle(x, y, 7, cols[k]) + t(x + 10, y + dy, f"{k} ({u:.2f}, {v:.2f})", 11, cols[k], weight="bold")
    b += t(370, 90, "donor = a + F·(b − c), F = 0.8", 13, weight="bold") + t(370, 112, "= (0.30, 0.30) + 0.8·(0.20, −0.10) = (0.46, 0.22)", 12, MUTED)
    b += t(370, 150, "Crossover (CR = 0.7): each gene from the donor", 13) + t(370, 170, "or the target → trial (0.46, 0.90)", 13)
    b += t(370, 208, "One-to-one selection: the trial replaces", 13) + t(370, 228, "the target only if it scores at least as well.", 13)
    svg("de.svg", 720, 340, b, "DE donor example")


def cmaes():
    x0, y0, S = 60, 50, 280
    b = t(30, 30, "CMA-ES: the search distribution learns the shape of the good region", 15, weight="bold") + unit_square(x0, y0, S, False)
    b += f"<path d='M{x0 + 20},{y0 + S - 20} L{x0 + S - 20},{y0 + 20}' stroke='{GREEN_SOFT}' stroke-width='38' stroke-linecap='round'/>"
    b += t(x0 + S - 30, y0 + 58, "good ridge", 11, GREEN, "end")
    for (cu, cv, a, bb, ang, op, lab) in [(.25, .3, 50, 50, 0, .35, "gen 1"), (.45, .5, 62, 30, -45, .55, "gen 4"), (.68, .72, 70, 16, -45, .9, "gen 8")]:
        x, y = P(x0, y0, S, cu, cv)
        b += f"<ellipse cx='{x:.1f}' cy='{y:.1f}' rx='{a}' ry='{bb}' transform='rotate({ang} {x:.1f} {y:.1f})' fill='none' stroke='{PURPLE}' stroke-width='2.5' opacity='{op}'/>"
        b += circle(x, y, 4, PURPLE) + t(x + 8, y + 22, lab, 11, PURPLE)
    b += t(380, 90, "Each generation CMA-ES samples from a normal", 13) + t(380, 110, "distribution N(m, σ²C), then:", 13)
    for i, s in enumerate(["moves the mean m towards the best samples,", "stretches the covariance C along directions", "that improved, and adapts the step size σ."]):
        b += t(380, 140 + i * 20, s, 12, MUTED)
    b += t(380, 222, "Here it learns that the two hyperparameters", 13) + t(380, 242, "must change together (a diagonal ridge).", 13)
    svg("cmaes.svg", 720, 360, b, "CMA-ES distribution adaptation")


def ucb():
    x0, y0, W, H = 70, 50, 560, 200
    N, c = 100, 0.1
    b = t(30, 30, "UCB1 exploration bonus c·√(ln N / n), c = 0.1, N = 100 evaluations", 15, weight="bold")
    b += line(x0, y0 + H, x0 + W, y0 + H, MUTED) + line(x0, y0, x0, y0 + H, MUTED)
    mx = 0.25
    pts = [(n, c * math.sqrt(math.log(N) / n)) for n in range(1, 101)]
    d = " ".join(("M" if i == 0 else "L") + f"{x0 + (n - 1) / 99 * W:.1f},{y0 + H - min(v, mx) / mx * H:.1f}" for i, (n, v) in enumerate(pts))
    b += f"<path d='{d}' fill='none' stroke='{GREEN}' stroke-width='3'/>"
    for v in (0.05, 0.10, 0.15, 0.20, 0.25):
        b += t(x0 - 8, y0 + H - v / mx * H + 4, f"{v:.2f}", 11, MUTED, "end")
    for n in (1, 25, 50, 75, 100):
        b += t(x0 + (n - 1) / 99 * W, y0 + H + 18, str(n), 11, MUTED, "middle")
    for n, lab in [(10, "PSO, n = 10: +0.068"), (40, "GA, n = 40: +0.034")]:
        v = c * math.sqrt(math.log(N) / n)
        x, y = x0 + (n - 1) / 99 * W, y0 + H - v / mx * H
        b += circle(x, y, 5, AMBER) + t(x + 8, y - 8, lab, 12, AMBER, weight="bold")
    b += t(x0 + W / 2, y0 + H + 38, "n = evaluations this family has used", 12, MUTED, "middle")
    b += t(30, 320, "Families that have used less of the budget get a larger bonus, so they are not written off too early.", 12, MUTED)
    svg("ucb.svg", 720, 338, b, "UCB1 exploration bonus")


# ------------------------------------------------------------------ 03
def tier_split():
    rnd = random.Random(5)
    b = arrowdef(MUTED) + t(30, 30, "Stratified tier split, N = 40 (darker = higher initial score)", 15, weight="bold")
    shade = lambda r: f"hsl(150, 40%, {78 - r * 1.3:.0f}%)"
    for i in range(40):
        b += circle(40 + i * 16, 70, 6, shade(39 - i))
    b += t(40, 96, "ranked best → worst, cut into 16 strata; one member drawn from each stratum goes to Tier 1", 12, MUTED)
    fams = ["GA", "PSO", "DE", "CMA-ES"]
    for k in range(4):
        x = 30 + k * 172
        b += r(x, 120, 160, 90, RED_SOFT, RED, 10) + t(x + 10, 140, f"Tier 1 · {fams[k]} (4)", 12, RED, weight="bold")
        ranks = sorted(rnd.sample(range(40), 4), reverse=True)
        for m, rk in enumerate([rnd.randint(b0 * 10, b0 * 10 + 9) for b0 in range(4)]):
            b += circle(x + 30 + m * 33, 180, 7, shade(39 - rk))
    for g in range(2):
        x = 30 + g * 344
        b += r(x, 232, 332, 96, PURPLE_SOFT, PURPLE, 10) + t(x + 10, 252, f"Tier 2 reserve, group {g + 1} (12)", 12, PURPLE, weight="bold")
        for m in range(12):
            rk = rnd.randint((m // 2) * 6, (m // 2) * 6 + 5)
            b += circle(x + 26 + (m % 6) * 50, 278 + (m // 6) * 30, 7, shade(39 - min(rk, 39)))
    b += t(30, 352, "Each island and group gets one member per rank block, so every family starts with a similar mix.", 12, MUTED)
    svg("tier_split.svg", 720, 370, b, "Stratified tier split")


def composite():
    rows = [("PSO", [.094, .25, .15, .12, .15], .061), ("GA", [.30, .091, .05, .054, .055], .061),
            ("CMA-ES", [.188, .136, .063, 0, .082], .061), ("DE", [0, 0, 0, .15, 0], .061)]
    names = ["fitness", "speed", "efficiency", "diversity", "trend", "UCB bonus"]
    cols = [GREEN, BLUE, AMBER, PURPLE, RED, "#9aa59f"]
    x0, W = 120, 520
    b = t(30, 30, "Composite score + UCB bonus per family (worked example, chapter 03 §4)", 15, weight="bold")
    for i, (fam, parts, bonus) in enumerate(rows):
        y = 60 + i * 52
        b += t(x0 - 10, y + 22, fam, 13, INK, "end", "bold")
        x = x0
        for v, col in zip(parts + [bonus], cols):
            w = v * W
            if w > 0:
                b += r(x, y, w, 32, col, rx=2)
            x += w
        b += t(x + 8, y + 21, f"UCB {sum(parts) + bonus:.3f}" + ("  promoted" if i < 2 else ""), 12, INK if i < 2 else MUTED, weight="bold" if i < 2 else "normal")
    for j, (n, col) in enumerate(zip(names, cols)):
        b += r(30 + j * 112, 280, 12, 12, col, rx=2) + t(46 + j * 112, 291, n, 11)
    b += t(30, 318, "Weights: fitness 0.30, speed 0.25, efficiency 0.15, diversity 0.15, trend 0.15 (each min-max normalised across the four).", 12, MUTED)
    svg("composite.svg", 720, 335, b, "Composite score breakdown")


def budget():
    segs = [("initial population", 40, "#9aa59f"), ("Tier 1: 5 iterations × 16", 80, RED), ("Tier 2: about 9 iterations × 40", 380, PURPLE)]
    x0, W = 30, 660
    b = t(30, 30, "Where 500 evaluations go in `evosynth run --full` (N = 40, t1_iters = 5)", 15, weight="bold")
    x = x0
    for lab, n, col in segs:
        w = W * n / 500
        b += r(x, 56, w, 40, col, rx=3) + t(x + w / 2, 81, str(n), 13, "#fff", "middle", "bold")
        x += w
    for i, (lab, n, col) in enumerate(segs):
        b += r(30, 120 + i * 26, 12, 12, col, rx=2) + t(48, 131 + i * 26, f"{lab}: {n}", 13)
    b += t(30, 222, "auto k: about 9 Tier 2 iterations → k = min(3, 9 // 2) = 3, so Tier 2 has 3 checkpoints.", 12, MUTED)
    b += t(30, 242, "Injections and rollbacks also spend budget; confirm-before-select (5 × 5 folds) does not.", 12, MUTED)
    svg("budget.svg", 720, 260, b, "Budget breakdown")


# ------------------------------------------------------------------ 05
def decode():
    names = ["colsample_bytree", "fe_strategy", "learning_rate", "max_depth", "model_type", "n_estimators", "reg_alpha", "subsample"]
    v = [.61, .9, .42, .55, .62, .58, .19, .66]
    dec = ["0.73", "interactions", "0.042 (log scale)", "7", "LightGBM (bin 0.50–0.75)", "310", "1.9", "0.83"]
    b = arrowdef(MUTED) + t(30, 30, "Decoding one latent vector into a configuration", 15, weight="bold")
    for i, (n, z, d) in enumerate(zip(names, v, dec)):
        y = 54 + i * 32
        b += t(30, y + 14, n, 12, MUTED) + r(170, y, 200, 18, GREY_SOFT, rx=9) + circle(170 + 200 * z, y + 9, 8, GREEN)
        b += t(382, y + 14, f"{z:.2f}", 12, INK) + line(420, y + 9, 470, y + 9, MUTED, 1.5, arrow=True) + t(480, y + 14, d, 12, INK, weight="bold")
    b += t(30, 330, "Categoricals use equal bins (min(⌊z·K⌋, K−1)); numbers are scaled to their range and integers rounded.", 12, MUTED)
    svg("decode.svg", 720, 348, b, "Latent vector decoding")


if __name__ == "__main__":
    for f in (split, cv, tree, bagging_boosting, overfitting, grid_random, overtuning, ga, pso, de, cmaes, ucb,
              tier_split, composite, budget, decode):
        f()
    print("figures written to", OUT)
