# EvoSynth — Complete Project Reference

**Full title:** *EvoSynth: Dynamic Multi-Armed Bandit Meta-Optimizer for Automated Model Selection and Hyperparameter Tuning*

**Project:** Group G002, Capstone / Project Phase 2 (course 23CSE498), Department of CSE.
**Team:** Shivam Kharat, Siddharth Ladda, Kaushal Loya, Paarthu Reddy, Priyansh Narang. **Guide:** Dr. Thangavelu S.
**Links:** GitHub `https://github.com/paarthureddy/EvoSynth` · Docs `https://evosynth-2027.github.io/evosynth-docs/` · Dataset (Zenodo) `https://doi.org/10.5281/zenodo.23645119`

This folder (`handbook/`, outside the official docs site) describes the whole EvoSynth project: its idea, design, code, experiments, results, paper and defence material. It is written so that someone can understand, explain, reproduce or rebuild the project from these files alone, without opening the repository.

It describes the code after the restructure (package `src/evosynth`, `settings/` folder, `evosynth` command, one results folder per run). The results are those of the panel-review (v1) runs, archived in `results/archive/2026-11-19_covertype_v1/`; they will be replaced after the Phase 0–5 fixes and re-runs.

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
2. **Initial population.** A random population is evaluated in parallel with a thread pool. Gradient-boosted models use early stopping.
3. **Two tiers.** The best 40% of the population is split equally among the four optimizers. These are the **Tier 1** islands. The other 60% is held in reserve.
4. **Tier 1 tournament.** All four optimizers evolve their islands for `t1_iters` iterations.
5. **Scoring and promotion.** A **composite score** combines fitness, speed, efficiency, diversity and trend. A **UCB1-style exploration bonus** is added to it. The top two optimizers are promoted to **Tier 2** and absorb the reserve.
6. **Main loop.** Tier 2 and the remaining Tier 1 optimizers keep running. Every `k` iterations a checkpoint can:
   - **SWAP** a surging Tier 1 optimizer with the weakest Tier 2 optimizer, under a grace period with rollback;
   - **inject** random vectors into stagnating islands;
   - **share elites** (the top 20%) between the Tier 2 islands.
7. **Output.** A single best configuration, that is one model plus its hyperparameters and feature engineering, not an ensemble. A post-search refit measures held-out accuracy.

Every step is streamed live to a FastAPI server-sent-events (SSE) server and shown on a React dashboard (the "glass box").

**Headline result** (UCI Forest Covertype, 581,012 rows, 54 features, 7 classes). Each configuration was searched on a 20k-row subsample with 2-fold CV, then refit on 60k rows and tested on 20k unseen rows.

| Budget | Search CV | Test accuracy | Winning configuration |
|---|---|---|---|
| 120 evaluations | 86.39% | 93.47% | ExtraTrees with `interactions` features |
| 300 evaluations | 86.39% | 93.82% | ExtraTrees with `interactions` features |
| 500 evaluations | 86.59% | 94.05% | ExtraTrees with `interactions` features |

Baselines on the same split:

| Baseline | Test accuracy |
|---|---|
| Default XGBoost | 80.97% |
| Default ExtraTrees | 89.59% |
| Optuna TPE | 89.80% |
| FLAML | 92.44% |
| Random Search (120 evaluations) | 93.92% |

The full tables and their caveats are in [08-results.md](08-results.md).

---

## What each file contains

| File | Contents |
|---|---|
| [README.md](README.md) | This index. A one-page summary of the project, a description of every file in the folder, and a glossary of EvoSynth terms. |
| [01-overview-and-motivation.md](01-overview-and-motivation.md) | The problem EvoSynth solves and why it matters. What each word of the title means and where it is implemented. The No Free Lunch argument. Why EvoSynth is not ensembling. The 7-stage workflow as text and as a Mermaid diagram. A written description of the architecture figure (`docs/public/arch.png`). The project objectives. |
| [02-architecture.md](02-architecture.md) | The complete repository layout, with every file and folder explained. Module and dependency structure. The runtime data flow from command to final output. The order of calls during one run. The run folder and the files written during a run. Threading model. Every constructor argument and the setting it comes from. |
| [03-engine-internals.md](03-engine-internals.md) | A line-level walkthrough of `DynamicOptimizer` (`controller.py`), Steps 0–6. Exact formulas and constants for composite scoring, diversity, trend and the UCB exploration bonus. Tier split arithmetic. SWAP, grace period, rollback, cooldown, stagnation injection and elite sharing. A worked example that reproduces the real scoring table from the Covertype log. The `UCB1Bandit` and `TierManager` helper classes and how far they are used. Where the documentation and the code disagree. |
| [04-optimizers.md](04-optimizers.md) | The ask/tell interface (`BaseOptimizer`). The exact implementation of GA, PSO, DE and CMA-ES: operators, constants, warm-start behaviour, selection and edge cases. A side-by-side comparison table. How to add a new optimizer. |
| [05-search-space-and-evaluators.md](05-search-space-and-evaluators.md) | All four ConfigSpace search spaces with ranges. How latent vectors are decoded (categorical binning, log scaling, dimension order). Every evaluator: the toy objective, `TabularEvaluator` (Covertype), and the PyTorch CNN for NAS. Their datasets, CV protocols and model construction rules. The `GenericFeatureEngineer` (`none` / `stats` / `interactions`). Early stopping. How to write a custom evaluator. |
| [06-api-and-dashboard.md](06-api-and-dashboard.md) | The FastAPI telemetry server (`/event`, `/stream`). The engine-side `EventReporter`. Every SSE event type with its payload and when it is emitted. The React/Vite/Tailwind dashboard: panels, state, what each view means. The VitePress documentation site and its deployment workflow. |
| [07-experiments.md](07-experiments.md) | Every experiment the project ran: what it tests, its exact configuration, dataset, budgets and protocol, and which command produces it. Covers the Covertype runs at 120/300/500 evaluations, the CVGS-300/500 ablations, the competitor baselines (default models, Random Search, Optuna, FLAML) and vision NAS. Also explains the staged orchestration script `tools/run_experiments.ps1`. |
| [08-results.md](08-results.md) | Every result in the repository as tables: the Covertype master comparison, the budget-scaling curve, the CVGS ablations, the live SWAP trace, and results reported only in the paper (speed-up, vision NAS, inference footprint). Interpretation of each result, plus notes on where each number comes from and how comparable it is. |
| [09-comparative-analysis.md](09-comparative-analysis.md) | EvoSynth compared with Optuna, FLAML, H2O AutoML, AutoGluon and Auto-sklearn. The ensembling-versus-search-time argument with its latency and memory reasoning. The full comparison matrix. Strengths and limitations. A list of related AutoML 2025/2026 papers and how each relates to EvoSynth. |
| [10-paper.md](10-paper.md) | The IEEE-format research paper (`paper/paper.tex`), section by section: abstract, method, algorithm pseudocode, experiments, tables, conclusion and references. The Panel Review slide deck (`paper/slides.tex`), slide by slide, including the team and the default parameter table. |
| [11-panel-defense-qa.md](11-panel-defense-qa.md) | The 30-second pitch, then likely evaluation and viva questions with answers grounded in the actual code and results. Covers novelty, datasets, mechanics, maths, performance, limitations, and the tricky questions the results invite. |
| [12-setup-and-usage.md](12-setup-and-usage.md) | Prerequisites and installation (Python, Node), dependencies, dataset caching, and how to run each `evosynth` command with its flags and `--set` overrides. Expected outputs, files produced and typical runtimes. Running the telemetry server and dashboard. Building the docs site. The staged experiment runner. Using EvoSynth from Python code. |
| [panel-defense-guide.md](panel-defense-guide.md) | The original Panel Review defence and viva guide (formerly `docs/PANEL_DEFENSE_MASTER_GUIDE.md`): elevator pitch, architecture talking points, formulas and prepared answers, as written before the review. Where it disagrees with the code, [03-engine-internals.md](03-engine-internals.md) §9 lists the differences. |

The files are numbered in reading order. Read 01 → 03 to understand the system, 07 → 08 for the evidence, and 11 to prepare a defence.

---

## Glossary

| Term | Meaning in EvoSynth |
|---|---|
| **Individual** | One candidate configuration. It holds a latent `vector` in `[0,1]^d`, a `fitness` (accuracy in %), and a `source_algorithm` tag such as `T2_PSO`. |
| **Latent space / Universal latent hypercube** | `[0,1]^d`, with one dimension per hyperparameter. Optimizers operate only on these vectors. Evaluators decode them into real configurations. |
| **Decode** | Turning a latent vector into hyperparameter values with ConfigSpace. Categorical dimensions are binned: `idx = min(floor(z*K), K-1)`. |
| **Evaluator** | An object with `.space` and `.evaluate(individual) -> float`. It trains a model and returns validation accuracy in percent. |
| **Budget** | The total number of evaluations (model trainings) allowed. It is shared by all optimizers, not multiplied by them. |
| **Tier 1 (T1)** | The "competition arena" or "sandbox". Holds 40% of the initial population, split into four islands, one per optimizer. After promotion it keeps the two optimizers that are not in Tier 2. |
| **Tier 2 (T2)** | The "working population". The two promoted optimizers, each with a larger island (its T1 island plus half of the 60% reserve). |
| **Island** | The population owned by one optimizer. |
| **Composite score** | A weighted sum of min–max-normalised metrics across the active optimizers: 0.30·fitness + 0.25·speed + 0.15·efficiency + 0.15·diversity + 0.15·trend. |
| **Diversity** | `min(std(fitnesses) / (max − min + 1e-9), 1.0)` over an island's fitness values. |
| **Trend / speed** | Improvement of an island's best fitness since the last scoring call: `max(best − prev_best, 0)`. |
| **UCB bonus** | `c · sqrt(ln(total_evals) / evals_of_algo)` with `c = 0.1`, added to the composite score to produce the "UCB score". |
| **Promotion** | After the Tier 1 tournament, the two optimizers with the highest UCB scores move to Tier 2. |
| **SWAP** | At a checkpoint, if the best T1 UCB score exceeds the worst T2 UCB score by more than 0.10, and cooldown and grace conditions allow it, the two optimizers trade tiers. |
| **Grace period / rollback** | After a SWAP, the promoted optimizer has 5 iterations to raise the global best. If it does not, the swap is reverted from a deep-copied snapshot. |
| **Cooldown** | The number of iterations (`2k`) during which an optimizer cannot be swapped back in. |
| **Stagnation / injection** | An island whose trend is 0 at a checkpoint is considered stagnant. T1 islands immediately have 50% replaced by random vectors. T2 islands have 20% replaced after 3 consecutive stagnant checkpoints. |
| **Elite sharing / migration** | At each checkpoint, the top 20% of each T2 island are pooled, and the best of the pool overwrite the bottom 20% of each T2 island. |
| **Checkpoint (k)** | Every `k_iters` iterations of a loop, when scoring, swapping, injection and sharing are evaluated. |
| **CVGS-300 / CVGS-500** | Covertype ablations with budgets of 300 and 500 and a population of 30. |
| **Isolated (ablation)** | One optimizer running alone on the full budget, with no tiers, sharing or swaps. Composite and UCB values are only recorded, never acted on. |
| **25% checkpoint** | In the ablations, each optimizer's state after 25% of the budget. This answers "what if each optimizer got an equal quarter of the budget?" |
| **FE / `fe_strategy`** | A searchable feature-engineering gene with values `none`, `stats` or `interactions`. |
| **Generalisation refit** | After the search, the winning configuration is retrained on 60k Covertype rows and tested on 20k unseen rows. |
| **SSE** | Server-sent events, the one-way HTTP stream from FastAPI to the dashboard. |
| **NFL** | The No Free Lunch theorem (Wolpert & Macready, 1997): no optimizer is best on all problems. |
| **CASH** | Combined Algorithm Selection and Hyperparameter optimisation, i.e. choosing the model family and its hyperparameters together. |
| **NAS** | Neural Architecture Search. Here, searching CNN channels, units, activation, dropout and the optimiser. |
