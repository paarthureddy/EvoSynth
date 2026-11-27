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

**Title:** *EvoSynth: Dynamic Multi-Armed Bandit Meta-Optimizer for Automated Model Selection and Hyperparameter Tuning* (the registered project title)
**Authors:** Shivam Kharat, Siddharth Ladda, Kaushal Loya, Paarthu Reddy, Priyansh Narang — Dept. of CSE, Amrita School of Computing, Amrita Vishwa Vidyapeetham, Coimbatore.
**Index terms:** automated machine learning, hyperparameter optimization, algorithm selection, multi-armed bandit, evolutionary algorithms, island model.

The paper follows the standard IEEE conference structure used in AutoML and evolutionary-computation venues: Introduction (with contributions) → Related Work → Method → Experimental Setup → Results → Discussion (with threats to validity) → Conclusion → Data Availability → References in order of first citation.

| Section | Content |
|---|---|
| Abstract | Problem (one optimizer chosen in advance), method (bandit-controlled GA/PSO/DE/CMA-ES portfolio, latent CASH encoding, composite reward, confirm before select), results (93.89% / 92.96% AUC, level with random search and Optuna, FLAML ±; ablation tie), released data |
| I. Introduction | HPO and CASH; optimizer families and No Free Lunch; algorithm selection as the framing; four contributions; paper outline |
| II. Related Work | A. HPO and CASH (SMAC, TPE, Optuna, Auto-WEKA, auto-sklearn, Hyperband, FLAML, overtuning); B. algorithm selection and portfolios (Rice, Kerschke et al., population-based algorithm portfolios, AMALGAM, adaptive operator selection, NGOpt); C. population-based and island-model search (island models, PBT, regularized evolution, NAS); D. AutoML systems and ensembles (H2O, AutoGluon) — each subsection ends with how EvoSynth differs |
| III. Method | Fig. 1; A. latent space; B. evaluation; C. population and tiers; D. composite reward; E. bandit control, swap, rollback; F. elite migration and injection; G. confirm, refit, measure; Algorithm 1 |
| IV. Experimental Setup | A. datasets and protocol; B. methods compared; C. ablation; D. implementation and reproducibility |
| V. Results | A. comparison with baselines (Table I); B. optimizer ablation (Table II, Fig. 2 convergence curves from `figures/make_paper_figures.py`); C. the Covertype plateau; D. meta-decisions in practice; E. speed-up (Table III) and inference cost; F. NAS on Fashion-MNIST |
| VI. Discussion | A. when the portfolio helps; B. threats to validity (one seed, plateau, unequal parallelism, single objective) |
| VII. Conclusion | summary and future work (multi-seed runs, HIGGS / Poker Hand, NGOpt and surrogate benchmarks, rank-normalised rewards, latency-aware objectives) |
| Data Availability | code, docs site, EvoSynth-Bench v2.1.0 DOI |

### References (47)
Every method, baseline, model, dataset and library the paper uses is cited. Textbook concepts (cross-validation, ROC AUC, stacking, CNNs) and datasets that appear only in future work (HIGGS, Poker Hand) are not. The bibliography is split into **Research Papers** (journal articles), **Conference Papers**, **Books and Book Chapters** and **Preprints**, each in order of first citation. It includes the ten recent (2024–2026) papers of the literature review: MaxUCB, RL-DAS, the MetaBBO survey, MsMA, Franceschi et al., Alcobaça & de Carvalho, overtuning, reshuffling, TabArena and carps. What each reference says and how EvoSynth uses it is in [11-papers.md](11-papers.md). The report lists 37 references, built the same way; the slides list only the ten recent papers.

---

## Part B — Review deck (`paper/slides.tex`)

**Title slide:** 23CSE498 Project Phase II (External Review), Team 2, Type 1 research project.
- Team: CB.SC.U4CSE23445 Shivam Kharat; CB.SC.U4CSE23565 Siddharth Ladda; CB.SC.U4CSE23627 Kaushal Loya; CB.SC.U4CSE23639 Paarthu Reddy; CB.SC.U4CSE23642 Priyansh Narang.
- Guide: **Dr. Thangavelu S.**, Associate Professor, Dept. of CSE. The next slide shows the guide's approval (`approval.png`).

| Slide | Content |
|---|---|
| 1. Problem and motivation | single fixed optimizer; No Free Lunch; Optuna / FLAML / H2O; gap; objectives: UCB1 meta-engine, live SWAPs, accuracy level with strong baselines under an identical held-out protocol from one model |
| 2. Workflow | The architecture figure `arch.png` only (the workflow-summary text block is kept in the source but commented out) |
| 3. Components | composite and UCB formulas; stagnation guard; Covertype and Adult; the split protocol; metrics: test score, search time, inference cost, ablation |
| 4. Implementation | module table including datasets, selection and refit, and the experiment harness |
| 5. Core modules | control flow; default parameters: budget 500, population 40, T₁ 5, k 3 (auto), c 0.1, swap margin 0.10, grace 5, elite 20%, injection 50% / 20%, 8 dimensions, confirmation top 5 with 5-fold CV |
| 6. Results | the two-dataset table (best untuned, FLAML, Random Search, Optuna, EvoSynth) and findings: tie with the strong baselines, right family per dataset, ablation, the Covertype SWAP and rollback |
| 7. Comparison | Optuna / FLAML / H2O / EvoSynth matrix with measured Covertype scores; single-model cost (0.26–12.5 ms); live demo with `evosynth run --full --dataset adult` |
| 8. Justification | NFL and the two tiers; latent space; 6.1× from parallel evaluation; rollback, injection and confirm-before-select; limitations (ties rather than wins, one seed, the Covertype plateau, single machine) and future scope |
| References | the ten recent (2024–2026) review papers only, grouped as Research Papers, Conference Papers and Preprints (see [11-papers.md](11-papers.md)) |

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
| 8. Deliverables | repository, docs site, EvoSynth-Bench v2.0.0; individual certifications: Shivam Kharat, Siddharth Ladda and Priyansh Narang — *Supervised Machine Learning in Python* (Udemy, 11 h); Kaushal Loya and Paarthu Reddy — *AI and Meta-Heuristics (Combinatorial Optimization) Python* (Udemy, 17.5 h) |

The deck follows the review template: it has no technical-enrichment slide; the individual Udemy certifications are listed in the report (§8). Images are looked up in `figures/`, `../` and `./`, so the deck builds both inside `paper/` and in the template's original folder layout.

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

**5.** What must be checked in the paper folder before the next submission?

<details><summary>Answer</summary>

That `approval.png` for that review is in `paper/figures/` (it changes for every review), that the numbers on the slides match `results/summary/main_table.md`, and that all three documents build.
</details>
