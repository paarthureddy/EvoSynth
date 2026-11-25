# 07 — Experiments

This file describes every experiment: what it tests, how it is configured, and which command produces it. The numbers are in [08-results.md](08-results.md). All values below are the defaults from `settings/` (mostly `settings/experiments.toml`); each can be changed there or per run with `--set`.

| # | Experiment | Dataset | Command | Run folder |
|---|---|---|---|---|
| E1 | Covertype demo + refit | UCI Covertype | `evosynth run [--full]` | `results/search/covertype/<preset>/seed_<k>/` |
| E2 | Covertype scaled runs (300, 500) | UCI Covertype | `evosynth scale --budget N` | `results/scaling/covertype/budget_<N>/seed_<k>/` |
| E3 | Competitor and floor baselines | UCI Covertype | `evosynth baselines --test ...` | `results/baselines/covertype/<baseline>/seed_<k>/` |
| E4 | CVGS-300 / CVGS-500 ablation | UCI Covertype | `evosynth ablation --setup CVGS-300` | `results/ablation/covertype/cvgs_<n>/seed_<k>/` |
| E5 | Vision NAS | sklearn digits | `evosynth nas` | `results/nas/digits/default/seed_<k>/` |
| E6 | Speed-up benchmark | Synthetic 50k × 25 | (reported in paper; no command) | Paper Table I |

Every run folder contains `manifest.json`, `result.json`, `run.log` and `events.jsonl` (see [02-architecture.md](02-architecture.md) §4). The Covertype experiments (E2–E4) are orchestrated by `tools/run_experiments.ps1` (§8). The panel-review (19 November) outputs of E1–E4 are archived unchanged in `results/archive/2026-11-19_covertype_v1/`; they were produced by the v1 code at git tag `v1-panel-review`.

---

## E1 — Covertype demo with refit (`evosynth run`)

- **Goal:** the headline real-world benchmark, covering model families, feature engineering and generalisation.
- **Search:** the 9-D space with `TabularEvaluator("covertype", subsample_size=20000)`, i.e. 2-fold CV on 20k stratified rows (`datasets.covertype.search_subsample`, `evaluation.cv_folds`).

| Preset | Budget | Pop | `t1_iters` | `k` |
|---|---|---|---|---|
| `search_demo` (default) | 120 | 20 | 3 | 2 |
| `search_full` (`--full`) | 500 | 40 | 5 | 3 |

  `--budget`, `--pop-size`, `--t1-iters`, `--k-iters` and `--subsample` override the preset; `--no-refit` skips Step 6.
- **Step 6 refit (the generalisation test, `models/refit.refit_and_score`):**
  1. Reload the full dataset and take a stratified split of **60,000 train / 20,000 test** (`refit.train_rows`, `refit.test_rows`, `run.random_state = 42`).
  2. Decode the winner and apply its FE strategy (fit on train, transform test).
  3. Build the winning model at full capacity (`models/factory.build_model(phase="refit")`):

| Family | Refit model |
|---|---|
| ExtraTrees / RandomForest | `n_estimators = max(n, 150)`, `max_depth = depth·3` if depth < 10 else unlimited, `max_features = colsample`, `n_jobs = −1` |
| LightGBM | `n_estimators = max(n, 250)` |
| XGBoost | `n_estimators = max(n, 200)`, `mlogloss` |

  4. Fit on 60k, then print the search CV accuracy, the test accuracy, the architecture and the FE strategy. `result.json` holds the best CV score, the best configuration, the winning algorithm and the refit numbers.

## E2 — Covertype scaled runs (`evosynth scale`)

- **Goal:** the budget-scaling curve (120 → 300 → 500) with crash-safe checkpointing.
- **Options:**

| Flag | Values | Default (`[experiments.scaling]`) |
|---|---|---|
| `--budget` | one of `budgets` = 120, 300, 500, 1000 | `default_budget` = 300 |
| `--pop-size` | integer | 30 |
| `--checkpoint-interval` | integer | 10 |
| `--variant` | folder name | `budget_<N>` |

- **Schedule:** `t1_iters = max(3, int(0.03·budget))` and `k = max(2, int(0.02·budget))`:
  - 300 evaluations → 9 / 6;
  - 500 evaluations → 15 / 10.
- **Fault tolerance:**
  - `evaluator.evaluate` is wrapped so that every 10 evaluations (and at the end) an **atomic** JSON checkpoint is written (temp file then `os.replace`) to `checkpoint.json` in the run folder;
  - the checkpoint holds `budget`, `evals_used`, `best_cv_accuracy`, `best_config`, `elapsed_seconds`, `status` and `timestamp`.
- **Refit:** the same 60k/20k refit as E1. The final checkpoint and `result.json` add the generalisation accuracy, the architecture and the FE strategy.
- **The 19 November run used** budget 300 and budget 500, both with pop 30; the 120 point of the curve is the E1 demo run (pop 20).

## E3 — Baselines (`evosynth baselines`)

All baselines use the **same 60k/20k split** (`random_state=42`). Each baseline gets its own run folder.

| Baseline (folder) | Search | Refit / test |
|---|---|---|
| Default ExtraTrees (`default_extratrees`) | None | `ExtraTreesClassifier(n_estimators=100)`, sklearn defaults, fit on 60k, test on 20k |
| Random Search (`random_search`) | 120 uniform random latent vectors scored with `TabularEvaluator(20k)` | Decode the best; refit **its model family and FE strategy** with `n_estimators=150` and library defaults (the tuned hyperparameters are **not** used) |
| Optuna TPE (`optuna_tpe`) | 120 trials of `TPESampler(seed=42)` over 9 `x_i ∈ [0,1]` floats, scored with `TabularEvaluator(20k)`; SQLite storage `optuna.db` with `load_if_exists` (resumable) | Same simplified refit as Random Search |
| FLAML AutoML (`flaml`) | `AutoML.fit` on the 60k training rows; `time_budget = 585 s` (about 10 min, matched to EvoSynth); `metric = accuracy`; estimators `lgbm`, `rf`, `extra_tree`, `xgboost`; holdout evaluation chosen by FLAML; seed 42 | FLAML's own final model, tested on 20k |

- **Options:** `--test` = `all` (default), `extratrees`, `random`, `optuna` or `flaml`; `--trials` (default 120); `--time-budget` (default 585).
- **Outputs per baseline:** `report.txt`, `result.json`, `run.log` (plus `optuna.db` / `flaml.log`).

**Default XGBoost / LightGBM / RandomForest (80.97 / 81.07 / 81.48%).** These floors appear in the paper, the slides and the v1 master table. The repository has no command that produces them yet (planned: Phase 4); they were obtained earlier and are reported as values.

## E4 — CVGS ablation on Covertype (`evosynth ablation`)

- **Question:** can any single optimizer, running alone, match EvoSynth's coordinated portfolio? And what if each optimizer gets an equal 25% share of the budget?
- **Setups** (`[experiments.ablation].setups`), evaluator `TabularEvaluator(20k)`:

| Setup | Budget | Pop | 25% checkpoint |
|---|---|---|---|
| CVGS-300 | 300 | 30 | 75 evaluations |
| CVGS-500 | 500 | 30 | 125 evaluations |

- **Isolated runs.** GA, PSO, DE and CMA-ES each run alone under strict rules:
  - each starts from its **own** random population of `pop_size`;
  - there is no co-evolution, no elite sharing and no swapping;
  - each run gets the full budget;
  - each run uses the same space, evaluator and seed policy.

  Mechanics:
  - Generation 0 evaluates `pop_size` random vectors in chunks.
  - The run then loops `ask(batch)` → evaluate → `tell`, with `batch = max(5, min(10, pop//3))`.
  - **Passive telemetry** is logged at every step: diversity, trend, speed `1/(t+1e-4)`, efficiency `best/ln(max(1,t)+e)`, composite (the `[composite]` weights with fixed scalings) and UCB (`c = 0.5`, `[experiments.ablation.passive_telemetry]`). It is **never used to make decisions**.
  - At the 25% checkpoint the state is captured: best accuracy, diversity and UCB.
- **EvoSynth run:** the same budget and population, with `t1_iters = max(3, int(0.04·budget))` and `k = max(2, int(0.02·budget))`. The checkpoint accuracy is read from `optimizer.history`. (Its diversity / composite / UCB values in the summary are still constants; measuring them is Phase 4.1.)
- **Resumable.** For each optimizer, a JSON (metadata and logs) and a pickle (best individual, population, all individuals) are saved atomically every 10 evaluations under `checkpoints/`. On restart, completed optimizers are skipped and interrupted ones resume from their last evaluation. `experiment_state.json` tracks the status of each algorithm. `max_workers = 2` by default to limit RAM.
- **Outputs:** `summary.txt` (Part 1: the 25% checkpoint; Part 2: full budget; plus a conclusion), `individual_runs/<SETUP>_<ALGO>.txt` (iteration log, checkpoint audit, final audit, discovered hyperparameters), `isolated_population.csv` (every evaluation of the isolated optimizers), `population.csv` (EvoSynth), `checkpoints/`, `result.json`.

## E5 — Vision NAS (`evosynth nas`)

- **Goal:** show that the engine is not limited to tabular data.
- **Setup:** `get_image_nas_search_space()` (7-D), `ImageCNNEvaluator(epochs=2, batch_size=64)`, budget 80, pop 20, `t1_iters = 3`, `k = 2` (`[experiments.nas]`). Dataset: scikit-learn digits (Fashion-MNIST replaces it in Phase 3).
- **Output:** the console log, the best CNN topology, `result.json`.

## E6 — Speed-up benchmark (paper Table I)

Compares three evaluation modes:
- serial (1 core);
- parallel batching (8 threads);
- parallel plus early stopping (8 threads).

It measures the initial 100-individual population and a full 120-evaluation run on a synthetic classification dataset (50,000 rows, 25 features, as described in the comparative analysis). No command in the repository reproduces it yet (planned: Phase 4, `measure_inference` / speed-up on Covertype).

---

## 8. `tools/run_experiments.ps1`, the staged orchestrator

```
param([string]$Stage = "menu", [string[]]$Extra = @())
```

- Runs the `evosynth` commands below with `py -3.14 -m evosynth`; `-Extra` arguments are passed to every command (e.g. `-Extra "--set","run.random_state=7"` for another seed).
- Outputs go to the usual run folders; re-running a stage overwrites that stage's folders, while the ablation and Optuna resume from their checkpoints.
- **Machine profile in the header:** Intel Core i5-12450HX (8 cores / 12 threads), 16 GB RAM.

| Stage | Command | Estimated duration |
|---|---|---|
| 1 Baselines | Installs optuna / flaml if missing, then `evosynth baselines --test all` | ~15–20 min (Random Search and Optuna took ~35 min each in practice) |
| 2 EvoSynth scaled | `evosynth scale --budget 300`, then `--budget 500` | ~1 h |
| 3 CVGS-300 | `evosynth ablation --setup CVGS-300` | Overnight |
| 4 CVGS-500 | `evosynth ablation --setup CVGS-500` | Overnight |
| A | Stages 1 → 4 in sequence, reporting total hours | — |

Usage: `.\tools\run_experiments.ps1` (interactive menu), or `.\tools\run_experiments.ps1 -Stage 2`.

The v1 `summary/` folder (`main_comparison_table.txt`, `convergence_curve.csv`) was compiled by hand from the stage outputs; generating summaries automatically (`build_summary`) is planned for Phase 4.
