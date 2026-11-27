# EvoSynth — Complete Project Reference

**Full title:** *EvoSynth: Dynamic Multi-Armed Bandit Meta-Optimizer for Automated Model Selection and Hyperparameter Tuning*

**Project:** Group G002, Capstone / Project Phase 2 (course 23CSE498), Department of CSE.
**Team:** Shivam Kharat, Siddharth Ladda, Kaushal Loya, Paarthu Reddy, Priyansh Narang. **Guide:** Dr. Thangavelu S.
**Links:** GitHub `https://github.com/paarthureddy/EvoSynth` · Docs `https://evosynth-2027.github.io/evosynth-docs/` · Dataset (Zenodo) `https://doi.org/10.5281/zenodo.23645119`

This folder (`handbook/`, outside the official docs site) describes the whole EvoSynth project: its idea, design, code, experiments, results, paper and defence material. It is written so that someone can understand, explain, reproduce or rebuild the project from these files alone, without opening the repository.

It describes the code after the fix plan (Phases 0–5: reproducible seeding, engine and optimizer fixes, real datasets with held-out test rows, one refit for every method). If you are new to machine learning or evolutionary algorithms, start with the three prerequisite chapters (00a–00c). Every chapter ends with a quiz. Chapters 08–10 and 13 report the Phase 5 smoke run (Covertype and Adult, seed 42); the panel-review (v1) results are archived in `results/archive/2026-11-19_covertype_v1/`.

---

## EvoSynth in one page

EvoSynth is an AutoML / hyperparameter-optimisation (HPO) framework written in Python. Most tuners run a single search algorithm. EvoSynth instead runs a **portfolio of four population-based optimizers**:

- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- Differential Evolution (DE)
- CMA-ES

All four search the same space at the same time. A **controller** decides which of them deserves more of the fixed evaluation budget.

How it works, in short:

1. **Universal latent space.** Every hyperparameter, including categorical choices such as the model family (`XGBoost`, `LightGBM`, `RandomForest`, `ExtraTrees`) and the feature-engineering strategy, is encoded as one coordinate in the unit hypercube `[0,1]^d`. Continuous optimizers can therefore move through mixed spaces.
2. **Initial population.** A random population is evaluated in parallel with a thread pool, with cross-validation on the dataset's search rows; gradient-boosted models stop early on a slice of the training fold.
3. **Two tiers.** A stratified split gives 40% of the population to four equal **Tier 1** islands, one per optimizer, each with a mix of good and weak members. The other 60% is held in reserve in two groups.
4. **Tier 1 tournament.** All four optimizers evolve their islands for `t1_iters` iterations.
5. **Scoring and promotion.** A **composite score** combines fitness, speed, efficiency, diversity and trend. The **UCB1 bandit** adds an exploration bonus and ranks the families; the top two are promoted to **Tier 2**, each taking one reserve group.
6. **Main loop.** Tier 2 and the remaining Tier 1 optimizers keep running. Every `k` iterations a checkpoint can:
   - **SWAP** a surging Tier 1 optimizer with the weakest Tier 2 optimizer, under a grace period with rollback;
   - **inject** random vectors into stagnating islands;
   - **share elites** pairwise (the top 20% each way) between the Tier 2 islands.
7. **Output.** The top 5 configurations are re-scored with 5-fold CV (confirm before select); the selected one, a single model with its hyperparameters and feature engineering, is refitted and scored once on held-out test rows, with its inference cost measured.

Every step is streamed live to a FastAPI server-sent-events (SSE) server and shown on a React dashboard (the "glass box").

**Results (Phase 5 smoke run, seed 42, 500 evaluations).** Every method is scored once on held-out test rows:

| Method | Covertype test accuracy | Adult test ROC AUC |
|---|---|---|
| Best untuned model | 89.77 | 92.92 |
| FLAML (same wall time) | 92.53 | **93.13** |
| Random Search | **93.89** | 92.98 |
| Optuna TPE | 93.87 | 92.95 |
| **EvoSynth** | **93.89** (12 min, the fastest) | 92.96 |

EvoSynth ties the strongest tuned baselines, picks the right model family for each dataset (ExtraTrees with interactions, XGBoost), and matches the best single optimizer in the ablation without knowing in advance which one it is. Parallel evaluation gives a measured 6.1× speed-up. The v1 numbers (up to 94.05%) came from runs with engine bugs and an unfair refit and are not comparable.

The full tables and their caveats are in [08-results.md](08-results.md).

---

## What each file contains

| File | Contents |
|---|---|
| [README.md](README.md) | This index. A one-page summary of the project, a description of every file in the folder, and a glossary of EvoSynth terms. |
| [00a-prerequisites-ml.md](00a-prerequisites-ml.md) | **Start here if you are new.** Machine-learning basics for EvoSynth: data, features, labels, classes; parameters vs hyperparameters (each of EvoSynth's 8); decision trees, forests and boosting; train / validation / test, cross-validation, overfitting; accuracy and ROC AUC; feature engineering; inference cost. |
| [00b-prerequisites-automl.md](00b-prerequisites-automl.md) | AutoML and hyperparameter search: search spaces, configurations, evaluations and budgets; grid, random and Bayesian search; FLAML and ensembles; No Free Lunch; overtuning, confirm before select, refit; seeds and surrogate benchmarks. |
| [00c-prerequisites-evolutionary.md](00c-prerequisites-evolutionary.md) | Evolutionary algorithms and bandits: GA, PSO, DE and CMA-ES with worked examples; the latent space; island models and elite sharing; multi-armed bandits and UCB1; portfolios and ask/tell. |
| [01-overview-and-motivation.md](01-overview-and-motivation.md) | The problem EvoSynth solves and why it matters. What each word of the title means and where it is implemented. The No Free Lunch argument. Why EvoSynth is not ensembling. The 7-stage workflow as text and as a Mermaid diagram. A written description of the architecture figure (`docs/public/arch.png`). The project objectives. |
| [02-architecture.md](02-architecture.md) | The complete repository layout, with every file and folder explained. Module and dependency structure. The runtime data flow from command to final output. The order of calls during one run. The run folder and the files written during a run. Threading model. Every constructor argument and the setting it comes from. |
| [03-engine-internals.md](03-engine-internals.md) | A line-level walkthrough of `DynamicOptimizer` (`controller.py`), Steps 0–6. Exact formulas and constants for composite scoring, diversity, trend and the UCB exploration bonus. Tier split arithmetic. SWAP, grace period, rollback, cooldown, stagnation injection and elite sharing. A worked example that reproduces the real scoring table from the Covertype log. The `UCB1Bandit` and `TierManager` helper classes and how far they are used. Where the documentation and the code disagree. |
| [04-optimizers.md](04-optimizers.md) | The ask/tell interface (`BaseOptimizer`). The exact implementation of GA, PSO, DE and CMA-ES: operators, constants, warm-start behaviour, selection and edge cases. A side-by-side comparison table. How to add a new optimizer. |
| [05-search-space-and-evaluators.md](05-search-space-and-evaluators.md) | All four ConfigSpace search spaces with ranges. How latent vectors are decoded (categorical binning, log scaling, dimension order). Every evaluator: the toy objective, `TabularEvaluator` (Covertype), and the PyTorch CNN for NAS. Their datasets, CV protocols and model construction rules. The `GenericFeatureEngineer` (`none` / `stats` / `interactions`). Early stopping. How to write a custom evaluator. |
| [06-api-and-dashboard.md](06-api-and-dashboard.md) | The FastAPI telemetry server (`/event`, `/stream`). The engine-side `EventReporter`. Every SSE event type with its payload and when it is emitted. The React/Vite/Tailwind dashboard: panels, state, what each view means. The VitePress documentation site and its deployment workflow. |
| [07-experiments.md](07-experiments.md) | Every experiment the project ran: what it tests, its exact configuration, dataset, budgets and protocol, and which command produces it. Covers search with refit, budget scaling (300/500/1000), the baselines (untuned models, Random Search, Optuna, FLAML), the CVGS-500/1000 ablation with the tier-split ablation, Fashion-MNIST NAS, the speed-up benchmark and the summary tables, plus the experiment runner `tools/run_experiments.py` and its plans. |
| [08-results.md](08-results.md) | The Phase 5 smoke run with charts: held-out test scores of every method on Covertype and Adult, search times, why every search finds the same Covertype model (the plateau), the CVGS-500 ablation, the engine's decisions in each run, the measured speed-up and inference cost, Fashion-MNIST NAS, and what changed since the v1 numbers. |
| [09-comparative-analysis.md](09-comparative-analysis.md) | EvoSynth compared with Optuna, FLAML, H2O AutoML, AutoGluon and Auto-sklearn, with the measured scores. The ensembling-versus-search-time argument with measured latency and size. The full comparison matrix. Strengths and limitations. Related AutoML 2025/2026 papers and how each relates to EvoSynth. |
| [10-paper.md](10-paper.md) | The IEEE-format paper (`paper/paper.tex`), the review report (`paper/report.tex`) and the review deck (`paper/slides.tex`), section by section and slide by slide, as updated with the Phase 5 results; how to build them. |
| [11-papers.md](11-papers.md) | All 47 references in the paper, report and slides, grouped by topic: the ten recent (2024–2026) papers of the literature review, then the foundations, optimizers, bandits and portfolios, island models, AutoML systems, models, datasets and software. For each: what it says, how EvoSynth uses it, and which documents cite it. How the bibliographies are grouped (Research Papers, Conference Papers, Books, Preprints) and what is deliberately not cited. |
| [12-setup-and-usage.md](12-setup-and-usage.md) | Prerequisites and installation (Python, Node), dependencies, dataset caching, and how to run each `evosynth` command with its flags and `--set` overrides. Expected outputs, files produced and typical runtimes. Running the telemetry server and dashboard. Building the docs site. The experiment runner. Using EvoSynth from Python code. Troubleshooting. |
| [commands.md](commands.md) | Every command with its options and defaults: setup, `evosynth run / scale / baselines / ablation / nas / speedup / summary / serve / settings / datasets`, the experiment runner, dashboard, docs and tests, and common recipes. |
| [13-panel-defense-qa.md](13-panel-defense-qa.md) | Everything for a review or viva in one place (it absorbs the former "Master Panel Defense Guide"): the pitch, key numbers, why these datasets, the data protocol, 27 questions with honest answers (including "is this overkill?"), the rival battlecard, strengths and limitations to state yourself, and a pre-review checklist. |
| [run-walkthrough.html](run-walkthrough.html) | An interactive, animated walkthrough of one run in 20 steps; at each step that can go wrong you choose success or a failure to see what happens. Open it in a browser. |
| [figures/](figures/) | The handbook's SVG figures and `make_figures.py`, which regenerates them. `make_results_figures.py` redraws the chapter 08 charts from `results/summary/`. |
| `htmlbuild/` | Builds the single-page HTML version of this handbook: `cd handbook/htmlbuild && npm install && npm run build` writes `handbook/evosynth-handbook.html` (git-ignored). |

The files are numbered in reading order. Newcomers read 00a → 00c first. Then 01 → 03 explain the system (with the run walkthrough alongside), 07 → 08 give the evidence, 10 → 11 cover the paper and the literature behind it, 12 plus Commands cover day-to-day use, and 13 prepares a defence.

---

## Glossary

| Term | Meaning in EvoSynth |
|---|---|
| **Individual** | One candidate configuration. It holds a latent `vector` in `[0,1]^d`, a `fitness` (its CV score in %, in the dataset's metric), and a `source_algorithm` tag such as `T2_PSO`. |
| **Latent space / Universal latent hypercube** | `[0,1]^d`, with one dimension per hyperparameter. Optimizers operate only on these vectors. Evaluators decode them into real configurations. |
| **Decode** | Turning a latent vector into hyperparameter values with ConfigSpace. Categorical dimensions are binned: `idx = min(floor(z*K), K-1)`. |
| **Evaluator** | An object with `.space` and `.evaluate(individual) -> float`. It trains a model and returns its cross-validated score in percent (accuracy or ROC AUC). |
| **Budget** | The total number of evaluations (model trainings) allowed. It is shared by all optimizers, not multiplied by them. |
| **Tier 1 (T1)** | The "competition arena". Holds 40% of the initial population, split by a stratified draw into four equal islands, one per optimizer. After promotion it keeps the two optimizers that are not in Tier 2. |
| **Tier 2 (T2)** | The "working population". The two promoted optimizers, each with a larger island (its T1 island plus one of the two reserve groups). |
| **Island** | The population owned by one optimizer. |
| **Composite score** | A weighted sum of min–max-normalised metrics across the active optimizers: 0.30·fitness + 0.25·speed + 0.15·efficiency + 0.15·diversity + 0.15·trend. |
| **Diversity** | Mean pairwise distance between an island's latent vectors, divided by √d (0 = identical, 1 = maximally spread). |
| **Speed / efficiency / trend** | Speed: improvement of the family's best since its last scoring call. Efficiency: that improvement per second of evaluation time. Trend: slope of the best over the last 3 scoring calls. |
| **UCB bonus** | `c · sqrt(ln(total_evals) / evals_of_algo)` with `c = 0.1`, added to the composite score by the `UCB1Bandit` to produce the "UCB score". |
| **Promotion** | After the Tier 1 tournament, the two optimizers with the highest UCB scores move to Tier 2. |
| **SWAP** | At a checkpoint, if the best T1 UCB score exceeds the worst T2 UCB score by more than 0.10, and cooldown and grace conditions allow it, the two optimizers trade tiers. |
| **Grace period / rollback** | After a SWAP, the promoted optimizer has 5 iterations to raise the global best. If it does not, the swap is reverted from a deep-copied snapshot. |
| **Cooldown** | The number of iterations (`2k`) during which an optimizer cannot be swapped back in. |
| **Stagnation / injection** | An island whose best did not improve since its last scoring call is stagnant. T1 islands immediately have 50% replaced by random vectors; T2 islands have 20% replaced after 3 consecutive stagnant checkpoints (through `optimizer.inject`). |
| **Elite sharing / migration** | At each checkpoint the two T2 islands trade elites pairwise: each one's worst 20% are replaced by the other's best 20%. |
| **Checkpoint (k)** | Every `k_iters` iterations of a loop, when scoring, swapping, injection and sharing are evaluated. |
| **CVGS-500 / CVGS-1000** | Ablations with budgets of 500 and 1000 and a population of 40 (v1 used CVGS-300 / CVGS-500 with population 30). |
| **Confirm before select** | Re-scoring the top 5 distinct configurations with 5-fold CV and selecting the best confirmed one, to counter noisy search scores. |
| **Test / search / train rows** | Each dataset's fixed split: test rows split off first and scored once at the end; search rows for cross-validation; train rows for the refit. |
| **Isolated (ablation)** | One optimizer running alone on the full budget, with no tiers, sharing or swaps. Composite and UCB values are only recorded, never acted on. |
| **25% checkpoint** | In the ablations, each optimizer's state after 25% of the budget. This answers "what if each optimizer got an equal quarter of the budget?" |
| **FE / `fe_strategy`** | A searchable feature-engineering gene with values `none`, `stats` or `interactions`. |
| **Generalisation refit** | After selection, the configuration is retrained on the dataset's train rows (60,000 for Covertype) and scored once on its held-out test rows (20,000). The same refit is used for every tuned method. |
| **SSE** | Server-sent events, the one-way HTTP stream from FastAPI to the dashboard. |
| **NFL** | The No Free Lunch theorem (Wolpert & Macready, 1997): no optimizer is best on all problems. |
| **CASH** | Combined Algorithm Selection and Hyperparameter optimisation, i.e. choosing the model family and its hyperparameters together. |
| **NAS** | Neural Architecture Search. Here, searching CNN channels, units, activation, dropout and the optimiser. |
