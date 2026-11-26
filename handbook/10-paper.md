# 10 — Paper, Report & Review Deck

The `paper/` folder holds:

| File | What it is |
|---|---|
| `paper.tex` | IEEE conference-format research paper |
| `report.tex` | Project Phase II (External Review) report in the Amrita format |
| `slides.tex` | Beamer deck for the review |
| `figures/` | `arch.png` (current architecture, generalised to any population N), `arch_legacy.png` (the panel-review diagram), `arch_source.html` (source of `arch.png`), `logo.png`. The slides also use `approval.png`, which is added fresh for each review and is not tracked |
| `links.md` | GitHub repository, documentation site, Zenodo dataset (`doi:10.5281/zenodo.23645119`) |
| `panel_review.pdf` | The rendered panel-review deck (local only, git-ignored) |

**Status.** All three documents describe the fixed engine (Phases 0–4) and report the Phase 5 smoke run ([08](08-results.md)): Covertype and Adult, one seed. The Wisconsin Breast Cancer and 30%-label-noise experiments of the panel-review paper are gone, as are the unmeasured 11.5× speed-up, the "2 ms / 150 MB" claim, and the v1 accuracy figures. Build any of them with `latexmk -pdf <file>.tex` inside `paper/` (MiKTeX, TeX Live or Tectonic), or upload the folder to Overleaf.

The Zenodo record (EvoSynth-Bench) is at **v2.0.0**: the eight real datasets' exact test / search / train row indices, a loader, and the v1 files kept for reference.

---

## Part A — The paper (`paper/paper.tex`)

**Title:** *EvoSynth: Dynamic Multi-Armed Bandit Meta-Optimizer for Automated Hyperparameter Tuning and Model Selection*
**Keywords:** HPO, AutoML, algorithm selection, multi-armed bandit, UCB1, GA, PSO, DE, CMA-ES, island model.

### Abstract (summary)
- **Problem:** no single optimizer is best everywhere, yet most tuners commit the whole budget to one.
- **Method:** a portfolio of GA, PSO, DE and CMA-ES in a two-tier island model, with budget reallocated online by a UCB1 bandit over a five-term composite score; model family, feature engineering and hyperparameters in one latent space; confirm before select, one refit, one score on held-out test rows.
- **Results:** 93.89% on Covertype and 92.96 ROC AUC on Adult with 500 evaluations, level with Random Search and Optuna; FLAML 1.4 lower on Covertype and 0.17 higher on Adult; matches the best single optimizer in the ablation; 6.1× measured speed-up from parallel evaluation; 81.8% on Fashion-MNIST NAS.

### I. Introduction
Hyperparameters and model choice (CASH); the drawbacks of grid/random search, single-strategy optimizers (No Free Lunch) and post-hoc ensembles. **Contributions:** (i) the two-tier bandit-controlled portfolio; (ii) one latent encoding of model, FE and hyperparameters; (iii) a protocol with fixed held-out test rows, confirm-before-select and an identical refit for every tuned method, published as EvoSynth-Bench v2.0.0; (iv) a measured comparison, ablation, speed-up benchmark and inference-cost measurement.

### II. System workflow (Fig. 1 = `arch.png`)
| Stage | Content |
|---|---|
| 1. Latent space | 8 dimensions: model family, FE strategy, learning rate, n_estimators, max_depth, subsample, colsample, L1. Linear/log decoding; categorical bins `min(⌊zK⌋, K−1)`; inactive genes are carried |
| 2. Population and split | N random vectors evaluated in parallel with 2-fold CV in the dataset's metric; stratified split: 4 Tier 1 islands of N/10 (40%), reserve in 2 groups (60%) |
| 3. Tier 1 | ask/tell for T₁ iterations; stagnant islands get 50% random members through `inject` |
| 4. Composite | `C = 0.30F + 0.25S + 0.15E + 0.15D + 0.15T`, each min–max normalised; D = mean latent distance / √d; T = slope over 3 scoring calls |
| 5. UCB1 control | `U = C + 0.1·sqrt(ln N_total / n_i)`; top 2 promoted; SWAP if a T1 family's U exceeds the weaker T2 family's by > 0.10; rollback after a 5-iteration grace period; cooldowns |
| 6. Tier 2 | pairwise elite sharing (each island's worst 20% replaced by the other's best 20%) |
| 7. Injection | 20% into a Tier 2 island stagnant for 3 checkpoints |
| 8. Confirm, refit, measure | top 5 distinct configurations re-scored with 5-fold CV; one refit on train rows; one test score; latency, throughput and size measured |

Algorithm 1 gives the lifecycle in pseudocode.

### III. Experimental evaluation
| Subsection | Content |
|---|---|
| Setup | Covertype and Adult, fixed splits; 10,000 search rows; 500 evaluations for EvoSynth, Random Search, Optuna; FLAML with equal wall time; untuned floors; one 12-thread CPU; seed 42 |
| Main comparison (Table I) | test score and search time of all eight methods on both datasets; the Covertype plateau explained |
| Ablation (Table II) | GA, PSO, DE, CMA-ES alone vs EvoSynth at 125 and 500 evaluations |
| Meta-decisions | promotion of DE + CMA-ES, GA SWAP and rollback on Covertype; two SWAPs on Adult; a different family wins each run |
| Speed-up (Table III) | 1.00× / 1.17× / 5.85× / 6.10× |
| Inference cost | 12.5 ms and 470 MB (Covertype ExtraTrees) vs 0.26 ms and 0.4 MB (Adult XGBoost) |
| NAS | Fashion-MNIST, 160 evaluations, 81.8% validation |

### IV. Dashboard
FastAPI SSE events (metrics, composite and UCB scores, tier assignments, every meta-decision) shown live in a React dashboard.

### V. Conclusion
EvoSynth matches strong baselines and the best single optimizer, picks the right model family, and is fastest on Covertype; it does not beat strong baselines on these datasets, FLAML wins Adult, and the Covertype space has a plateau. **Future work:** multiple seeds and harder datasets, NGOpt and surrogate benchmarks, rank-normalised rewards, latency-aware objectives.

### References (20)
Feurer & Hutter 2019; Bergstra & Bengio 2012; Auer et al. 2002; Kennedy & Eberhart 1995; Storn & Price 1997; Hansen & Ostermeier 2001; Chen & Guestrin 2016; Ke et al. 2017; Fialho et al. 2010; Bergstra et al. 2011; Wang et al. 2021 (FLAML); LeDell & Poirier 2020 (H2O); Zoph & Le 2017; Real et al. 2019; Hutter et al. 2011 (SMAC); Wolpert & Macready 1997; Akiba et al. 2019 (Optuna); Blackard & Dean 1999 (Covertype); Kohavi 1996 (Adult); Xiao et al. 2017 (Fashion-MNIST).

---

## Part B — Review deck (`paper/slides.tex`)

**Title slide:** 23CSE498 Project Phase II (External Review), Team 2, Type 1 research project.
- Team: CB.SC.U4CSE23445 Shivam Kharat; CB.SC.U4CSE23565 Siddharth Ladda; CB.SC.U4CSE23627 Kaushal Loya; CB.SC.U4CSE23639 Paarthu Reddy; CB.SC.U4CSE23642 Priyansh Narang.
- Guide: **Dr. Thangavelu S.**, Associate Professor, Dept. of CSE. The next slide shows the guide's approval (`approval.png`).

| Slide | Content |
|---|---|
| 1. Problem and motivation | single fixed optimizer; No Free Lunch; Optuna / FLAML / H2O; gap; objectives: UCB1 meta-engine, live SWAPs, accuracy level with strong baselines under an identical held-out protocol from one model |
| 2. Workflow | `arch.png`; encode → stratified 40/60 split → UCB1 promotes top 2 → Tier 2 with pairwise elite sharing, SWAP and rollback → confirm top 5 → one refit → one test score |
| 3. Components | composite and UCB formulas; stagnation guard; Covertype and Adult; the split protocol; metrics: test score, search time, inference cost, ablation |
| 4. Implementation | module table including datasets, selection and refit, and the experiment harness |
| 5. Core modules | control flow; default parameters: budget 500, population 40, T₁ 5, k 3 (auto), c 0.1, swap margin 0.10, grace 5, elite 20%, injection 50% / 20%, 8 dimensions, confirmation top 5 with 5-fold CV |
| 6. Results | the two-dataset table (best untuned, FLAML, Random Search, Optuna, EvoSynth) and findings: tie with the strong baselines, right family per dataset, ablation, the Covertype SWAP and rollback |
| 7. Comparison | Optuna / FLAML / H2O / EvoSynth matrix with measured Covertype scores; single-model cost (0.26–12.5 ms); live demo with `evosynth run`; Fashion-MNIST NAS 81.8% |
| 8. Justification | NFL and the two tiers; latent space; 6.1× from parallel evaluation; rollback, injection and confirm-before-select; limitations (ties rather than wins, one seed, the Covertype plateau, single machine) and future scope |
| Enrichment | MOOCs: Kharat, Ladda, Loya — *Evolutionary Computation and its Applications* (Coursera); Reddy — *Front-End Web Development with React* (Coursera); Narang — *Optimization for ML* (NPTEL) |
| References | the ten review references (Scrivano 2025 … LeDell & Poirier 2020) |

The deck ends with "Thank You — Questions & Viva Voce".

---

## Part C — Review report (`paper/report.tex`)

Two to three pages in the Amrita format (maroon headings, team and guide table):

| Section | Content |
|---|---|
| Abstract | as the paper, with both datasets and the measured inference cost |
| 2. Problem, motivation, literature | problem statement; objectives (bandit allocation, two-tier architecture, fair and reproducible evaluation); literature table (Scrivano, Alcobaça & Carvalho, FLAML, H2O, Crepinšek et al.) and the research gap |
| 3. Methodology | `arch.png`; latent space; composite score; UCB1; tiers, SWAP and rollback; the 8-dimensional search space; confirm before select |
| 4. Implementation | environment and repository; module table (latent space, optimizers, UCB1 engine, tier manager, feature engineer, datasets and evaluators, selection and refit, experiment harness, telemetry); feature engineering; logging and reproducibility; the held-out workflow |
| 5. Results | the two-dataset table; findings; ablation; the Covertype SWAP, rollback and winner; Fashion-MNIST NAS |
| 6. Justification | algorithmic rationale; complexity; single model with measured cost; edge cases (rollback, injection, confirm before select, failed folds) |
| 7. Conclusion | conclusion, limitations, future scope |
| 8. Deliverables | repository, docs site, EvoSynth-Bench v2.0.0, individual certifications |

> **Check before submitting:** the certification lists in `report.tex` (Udemy courses) and `slides.tex` (Coursera / NPTEL courses) differ. Make them match whatever each member actually completed.

---

## Quiz

**1.** Which experiments of the panel-review paper were removed, and why?

<details><summary>Answer</summary>

The Wisconsin Breast Cancer and 30%-label-noise synthetic experiments. The project moved to real benchmark datasets with fixed held-out splits (Covertype, Adult and six more), and the code no longer contains those experiments.
</details>

**2.** What does the paper now claim about EvoSynth's accuracy against Random Search and Optuna?

<details><summary>Answer</summary>

That it is level with them (within 0.03 points on both datasets) under an identical protocol, not better. Its claimed advantages are choosing the right model family, matching the best single optimizer without choosing it in advance, and a shorter wall time from parallel evaluation.
</details>

**3.** Name the three tables of the paper's evaluation section.

<details><summary>Answer</summary>

Table I, held-out test score and search time of every method on Covertype and Adult; Table II, the optimizer ablation (each optimizer alone vs EvoSynth); Table III, the speed-up benchmark.
</details>

**4.** Which figure replaced the panel-review architecture diagram, and what changed in it?

<details><summary>Answer</summary>

The new `arch.png`. It is generalised to any population size N (four Tier 1 islands of N/10, a 60% reserve in two groups), shows the stratified split and pairwise elite sharing, and ends with confirm before select, one refit and a held-out test score. The old diagram is kept as `arch_legacy.png`.
</details>

**5.** What must be checked in the report and slides before the next submission?

<details><summary>Answer</summary>

That the individual certification lists match (the report lists Udemy courses and the slides list Coursera and NPTEL courses), and that `approval.png` for that review is in `paper/figures/`.
</details>
