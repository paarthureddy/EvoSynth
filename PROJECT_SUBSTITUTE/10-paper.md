# 10 — Research Paper & Panel Review Deck

The `submission/` folder holds:

| File | What it is |
|---|---|
| `G002_Paper.tex` | IEEE conference-format paper |
| `maint.tex` | Beamer slide deck for the Panel Review |
| `Group2___PPT__Panel_Review___2_ (1).pdf` | The rendered review deck |
| `G002_Github.url` | Link to the GitHub repository |
| `G002_Website.url` | Link to the documentation site |
| `G002_Dataset.url` | Link to the Zenodo dataset (`doi:10.5281/zenodo.23645119`) |

The Zenodo dataset is "EvoSynth-Bench". It contains the pathological classification benchmark CSV and the HPO trajectory CSV, which can be checked with `scripts/load_dataset.py`.

---

## Part A — The paper (`G002_Paper.tex`)

**Title:** *EvoSynth: Dynamic Multi-Armed Bandit Meta-Optimizer for Automated Hyperparameter Tuning and Model Selection*
**Byline:** Group G002, Dept. of CSE (Capstone Engineering Research Project).
**Keywords:** HPO, AutoML, Multi-Armed Bandit, UCB1, GA, PSO, DE, CMA-ES, Cooperative Co-evolution.

### Abstract (summary)

**Problem:** single-algorithm tuners get trapped in local minima or waste evaluations on flat regions, and ensembles add inference latency and memory.

**Proposal:** EvoSynth runs DE, PSO, CMA-ES and GA at the same time inside a two-tier island model controlled by a UCB1 multi-armed bandit. Its stages are:
1. map all hyperparameters and model choices into `[0,1]^d`;
2. split the population into a 40% Tier 1 arena and 60% reserve;
3. run an exploration tournament;
4. score optimizers with a composite metric (accuracy, speed, diversity, trend);
5. promote the top optimizers to Tier 2 using UCB1 exploration bonuses;
6. share elites periodically;
7. inject diversity on stagnation.

Evaluation is parallel across CPU cores, with early stopping for boosted trees.

**Reported results:**
- 11.5× speed-up in population evaluation;
- recovery from deceptive noisy local minima;
- dynamic model switching across XGBoost, LightGBM and RandomForest;
- CNN architecture tuning reaching 95.33%.

### I. Introduction

Hyperparameters cannot be learned by gradient descent and strongly affect performance. Three conventional approaches each fall short:
1. **Grid and random search:** grid search grows exponentially; random search ignores feedback.
2. **Single-algorithm heuristic search:** no single method suits every dataset.
3. **Heavy ensembling:** adds production latency.

**The EvoSynth approach.** A meta-optimizer over four complementary algorithms:
- DE explores through vector differences;
- PSO accelerates using personal and global bests;
- CMA-ES learns covariance and fine-tunes;
- GA explores globally through selection, crossover and mutation.

The UCB1 bandit gives more budget to whichever optimizer is progressing, solutions are shared between optimizers, and the output is one lean model.

### II. System workflow and architecture (Fig. 1 = `arch.png`)

| Stage | Content |
|---|---|
| 1. Input layer and universal latent space | Mixed types are normalised to `z ∈ [0,1]`. Continuous values: `x = L + z(U − L)` (or log scale). Categorical values: `Index = min(⌊zK⌋, K − 1)`. Example with K = 3 for `model_type`: [0, .33) → XGBoost, [.33, .66) → RF, [.66, 1] → LightGBM. |
| 2. Initialisation and allocation | N = 100 random individuals are evaluated by CV. Tier 1 gets 40 (10 per optimizer) and Tier 2 holds 60 in reserve. |
| 3. Tier 1 arena | Each optimizer runs `T1` iterations (typically 10–30) under the ask/tell lifecycle. |
| 4. Composite scoring | `Composite = w_f·Fitness + w_s·Speed + w_d·Diversity + w_t·Trend`, with **w_f = 0.40, w_s = w_d = w_t = 0.20**. Speed = `max(Best_t − Best_{t−1}, 0)`. Diversity = `min(σ/(max − min + ε), 1)`. Trend = a multi-interval moving average. *(The code uses 5 metrics weighted 0.30/0.25/0.15/0.15/0.15; see [03](03-engine-internals.md) §9.)* |
| 5. Control layer | Tier comparator, stagnation detector, UCB1 selector (`Score_i = Composite_i + c·sqrt(ln N_total / n_i)`, c = 0.1), and a promotion/demotion engine that promotes the top two and gives them the reserve. |
| 6. Tier 2 and elite sharing | Each island has 10 + 30 = 40 individuals. Every k iterations (5 or 10), the top 20% of both islands are pooled, sorted, and copied over each island's worst individuals. |
| 7. Diversity injection | T1 batch refresh (bottom 50% replaced), T1 global reset (keep the top x%), T2 island injection (bottom 20% replaced). |
| 8. Parallelism and early stopping | `ThreadPoolExecutor` (boosting libraries release the GIL); boosted models stop after 10 non-improving rounds. |

**Algorithm 1: EvoSynth lifecycle**

```
Require: space S, evaluator E, budget B, pop N, T1 iterations, sharing interval k
P ← random population (N); evaluate in parallel
split P → P_T1 (40%), P_reserve (60%); seed DE, PSO, CMA-ES, GA with P_T1
for t = 1..T1: for each algo: Z ← algo.ask(|P_T1|/4); evaluate Z (parallel, early stopping); algo.tell(Z)
Scores ← CompositeScoringEngine(T1 algos); Top2, Bot2 ← UCB1(Scores, c = 0.1)
promote Top2 to Tier 2 and inject P_reserve
while evals < B:
    step Tier 2 and active Tier 1
    if iter mod k == 0: share top 20% elites between T2 islands; if stagnation: inject random vectors into bottom 20%
return best architecture and hyperparameters
```

### III. Experimental evaluation

**Setup:**
- Wisconsin Breast Cancer: 569 × 30, binary.
- Pathological noisy dataset: 2,000 × 20 with `flip_y = 0.3`.
- Handwritten digits for NAS: 1,797 images of 8×8.

**Table I — speed-up:**

| Mode | Cores | Initial pop (s) | Total (s) |
|---|---|---|---|
| Sequential | 1 | 44.8 | 392.4 |
| Parallel batching | 8 threads | 5.1 | 78.2 |
| **EvoSynth (parallel + early stopping)** | 8 threads | **3.9** | **63.9** |

Initial population evaluation is **11.5× faster**. A full 120-evaluation run drops from more than 6.5 min to 63.9 s.

**Table II — UCI Forest Covertype** (581,012 × 54, 7 classes; classes 1 and 2 make up more than 84% of the data, so defaults reach about 81% largely through the majority classes):

| Method | Model | FE | Time | Test accuracy |
|---|---|---|---|---|
| Default 1 | XGBoost | none | 2.1 s | 80.97 |
| Default 2 | LightGBM | none | 1.2 s | 81.07 |
| Default 3 | RandomForest | none | 3.4 s | 81.48 |
| EvoSynth (raw features) | XGBoost (DART, depth 10) | none | 6 min 14 s | 82.73 |
| EvoSynth (search CV) | ExtraTrees (334 trees, depth 8) | interactions | 8 min 30 s | 86.39 |
| **EvoSynth (generalisation)** | **ExtraTrees (60k / 20k)** | **interactions** | **1 min 14 s** | **93.47** |

- The search covered 4 model families and 3 FE strategies in a 9-D space.
- The `interactions` features (`c_i − c_j` and `sqrt(c_i² + c_j²)`) combined with ExtraTrees cut error from 19.0% to 6.53%, a 64.6% relative reduction, in 9 min 44 s in total.
- The paper also describes the **live SWAP**: GA's diversity fell to 0.450 and its UCB to 0.547, while Tier 1 PSO held diversity 0.500 and reached UCB 0.720. PSO was promoted and GA demoted.

**Noisy local minima:**
- Static PSO and hill-climbing plateaued around 76%.
- EvoSynth's stagnation detection, exploration bonus, elite reseeding and 20% injection reached **79.40%**.

**Table III — multi-model output on the noisy data:**

| Parameter | Range | Selected |
|---|---|---|
| Model | XGB / LGBM / RF | **XGBoost** |
| learning_rate | [0.01, 0.3] (log) | 0.0163 |
| n_estimators | [50, 500] | 327 |
| max_depth | [3, 10] | 6 |
| subsample | [0.5, 1.0] | 0.9797 |
| colsample_bytree | [0.3, 1.0] | 0.9270 |
| reg_alpha | [0, 10] | 7.5669 |
| booster | gbtree / dart | gbtree |

Strong L1 regularisation together with subsampling suppressed overfitting to the noisy labels.

**Vision NAS** (`ImageCNNEvaluator`):
- 80 evaluations in 14.6 s reached **95.33%**.
- Best topology: conv1 = 16, conv2 = 64, fc = 64, dropout 0.0107, LeakyReLU, Adam with lr 0.0309.

### IV. Real-time glass-box dashboard

FastAPI SSE (`/stream`) emits `metric`, `composite`, `tier_state` and `switch` events. A React + Vite + Tailwind dashboard shows the accuracies, the tier assignments, the composite breakdowns and the mirrored logs live.

### V. Conclusion

EvoSynth projects mixed hyperparameters into a universal latent space and manages DE, PSO, CMA-ES and GA in a two-tier island model governed by UCB1 composite scoring. This lets it overcome local minima without multiplying inference latency. Its features are:
- parallel evaluation;
- early stopping;
- diversity injection;
- tabular model switching;
- vision NAS.

**Results:** an 11.5× speed-up, escape from noisy local optima, and full observability.

### References (15)

1. Feurer & Hutter 2019 (HPO chapter)
2. Bergstra & Bengio 2012
3. Auer, Cesa-Bianchi & Fischer 2002 (UCB1)
4. Kennedy & Eberhart 1995
5. Storn & Price 1997
6. Hansen & Ostermeier 2001
7. Chen & Guestrin 2016
8. Ke et al. 2017
9. Fialho et al. 2010
10. Bergstra et al. 2011
11. Wang et al. 2021 (FLAML)
12. LeDell & Poirier 2020 (H2O)
13. Zoph & Le 2017
14. Real et al. 2019
15. Hutter, Hoos & Leyton-Brown 2011 (SMAC)

---

## Part B — Panel Review deck (`maint.tex`)

**Title slide:**
- Team (S.No, register number, name):
  1. CB.SC.U4CSE23445 — Shivam Kharat
  2. CB.SC.U4CSE23565 — Siddharth Ladda
  3. CB.SC.U4CSE23627 — Kaushal Loya
  4. CB.SC.U4CSE23639 — Paarthu Reddy
  5. CB.SC.U4CSE23642 — Priyansh Narang
- Guide: **Dr. Thangavelu S.**, Associate Professor, Dept. of CSE.
- The next slide shows the guide's approval email (`approval.png`).

**1. Problem definition and motivation**
- **Problem:** a single fixed optimizer stagnates on non-convex landscapes.
- **Context:** the No Free Lunch theorem.
- **Existing work:**
  - Optuna: single Bayesian sampler;
  - FLAML: cost-frugal greedy search;
  - H2O: sequential Bayesian search plus 10-model stacking.
- **Gap:** no framework runs concurrent optimizers with live stagnation detection and dynamic swaps while producing a single model.
- **Objectives:**
  1. a UCB1 meta-engine;
  2. live tier SWAPs;
  3. ensemble-level accuracy (93.47%) from one model (2 ms).

**2. Methodology and workflow**
- Figure `arch_updated.png`.
- Summary: encode to `[0,1]^d` → 40% Tier 1 tournament → UCB1 selects the top two → 60% Tier 2 with elite sharing → decode and evaluate (XGB / LGBM / RF / ET + generic FE) → refit on 60k → single model.

**3. Components and formulation**
- UCB1 meta-engine with a 5-metric composite score.
- Two-tier island model.
- Stagnation guard: zero trend triggers injection; a T1 optimizer leading by more than 10% triggers a SWAP.
- `UCB_i = CS_i + c·sqrt(ln E_total / E_i)` with c = 0.1.
- `CS = 0.30F + 0.25S + 0.15E + 0.15D + 0.15T`.
- **Data:** Covertype; 20k search subsample (the deck says 3-fold CV; the code uses 2-fold); 60k/20k refit; FE options none / stats / interactions.
- **Metrics:** test accuracy, relative error reduction `(e_base − e_ours)/e_base` (64.6%), wall-clock time.

**4. Implementation overview.** All modules are marked 100%:

| Module | Status |
|---|---|
| Latent mapping | Integrated |
| UCB1 meta-engine (scoring, bonus, promotion, swap + rollback) | Integrated |
| Ask/tell wrappers for the optimizers | Integrated |
| Tier manager (40/60, elite sharing, injection) | Integrated |
| Evaluators (4 families + FE + early stopping) | Functional |
| Generic feature engineer | Integrated |
| SSE → React dashboard | Integrated |

- **Environment:** Python 3.14, XGBoost, LightGBM, scikit-learn, PyTorch and React, on a laptop CPU with no GPU.
- **Repository:** GitHub branch `PostReview2`, plus the docs site and Zenodo.

**5. Core module details.** Control flow: init → Tier 1 for 10 iterations → promotion → live SWAP (lead greater than 10%) → rollback (must beat the global best within 5 iterations) → top-20% elite sharing. FE is a searchable gene that needs no domain knowledge.

Default parameters listed on the slide:

| Parameter | Value |
|---|---|
| Evaluation budget | 120 (Covertype) |
| Population | 30 |
| Tier 1 / Tier 2 split | 40% / 60% |
| Tier 1 iterations | 10 |
| Checkpoint k | 5 |
| c | 0.1 |
| Swap threshold | 10% |
| Rollback grace period | 5 iterations |
| Elite share | Top 20% |
| Injection (T1 / T2) | 50% / 20% |
| Dimensions | 9 |
| Model families | 4 |

(The actual 120-evaluation run used pop 20, 3 Tier 1 iterations and k = 2.)

**6. Verification and quantitative results**

| Method | Model | Time | Accuracy |
|---|---|---|---|
| Default XGBoost | Untuned | 2 s | 80.97 |
| Default LightGBM | Untuned | 1 s | 81.07 |
| Default RandomForest | Untuned | 3 s | 81.48 |
| EvoSynth search (20k) | ExtraTrees + FE | 8 min 30 s | 86.39 |
| **EvoSynth final (60k/20k refit)** | **ExtraTrees** | **9 min 44 s** | **93.47** |

- This is a 64.6% relative error reduction.
- **SWAP confirmed:** at iteration 2, T1_PSO (UCB 0.720, diversity 0.500) replaced T2_GA (UCB 0.547, diversity 0.450).

**7. Comparison and demonstration**

| Aspect | Optuna | FLAML | H2O | EvoSynth |
|---|---|---|---|---|
| Search paradigm | Bayesian | Cost-frugal | Random/Bayes | UCB1 bandit |
| Strategies | 1 sampler | 1 greedy search | Sequential | 4 concurrent |
| Switching | None | None | None | Tier SWAPs |
| Diversity protection | None | None | None | Diversity in UCB |
| Feature engineering | Manual | Limited | Fixed | Dataset-agnostic gene |
| Output | 1 model | 1 model | 10-model stack | 1 model (2 ms, 150 MB) |

- **Pareto advantage:** 93.47% without the cost of an ensemble.
- **Live demo:** `run_covertype.py` with the dashboard.
- **Vision NAS:** **95.78%** in 80 evaluations (17.7 s).

**8. Technical justification and trade-offs**
- **Justification:** NFL, plus a two-tier separation of exploration and exploitation.
- **Trade-offs:**
  - latent decoding adds overhead but makes categoricals tractable for PSO and CMA-ES;
  - meta-logic overhead is under 0.05% of runtime;
  - the meta-optimisation is offline, so there is no inference penalty.
- **Boundary conditions:** the rollback guard, stagnation rescue, and robustness under 30% label noise (about 79.4%).
- **Limitations:** single machine; FE designed for tabular data.
- **Future work:** Ray-distributed workers, warm-start meta-learning, deep-learning NAS.

**Technical enrichment (MOOCs, 10 h each):**

| Member | Course | Platform |
|---|---|---|
| Kharat, Ladda, Loya | Evolutionary Computation and its Applications | Coursera |
| Reddy | Front-End Web Development with React (dashboard) | Coursera |
| Narang | Optimization for ML (mathematical latent space) | NPTEL |

**References (10):**
1. Scrivano 2025 — AutoML trends
2. Alcobaça & de Carvalho 2025 — AutoML review
3. Estevanell-Valladares et al. 2025 — XAutoLM
4. Borboudakis et al. 2024 — meta-level space reduction
5. Crepinšek et al. 2025 — MsMA stagnation
6. Gerber et al. 2025 — online meta-learning AutoML
7. Auer et al. 2002 — UCB1
8. Wolpert & Macready 1997 — No Free Lunch
9. Wang et al. 2021 — FLAML
10. LeDell & Poirier 2020 — H2O

The deck ends with "Thank You — Questions & Viva Voce".
