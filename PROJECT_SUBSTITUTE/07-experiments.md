# 07 — Experiments

This file describes every experiment: what it tests, how it is configured, and which script produces it. The numbers are in [08-results.md](08-results.md).

| # | Experiment | Dataset | Script | Output |
|---|---|---|---|---|
| E1 | Pathological demo | Noisy synthetic (30% flips) | `main.py` → `scripts/main.py` | Console + `results/run_output.txt`, `results/csv/` |
| E2 | UPGS-120 / UPGS-2500 ablation | Noisy synthetic | `scripts/run_ablation_study.py` | `results/ablation_*`, `results/individual_runs/` |
| E3 | Covertype demo + refit | UCI Covertype | `run_covertype.py` → `scripts/run_covertype.py` | Console (`results/covertype_run_with_fe.txt`, `results_forest/.../run_120.txt`) |
| E4 | Covertype scaled runs (300, 500) | UCI Covertype | `scripts/run_covertype_evosynth.py` | `results_forest/<run>/evosynth/` |
| E5 | Competitor and floor baselines | UCI Covertype | `scripts/run_baselines_forest.py` | `results_forest/<run>/baselines/` |
| E6 | CVGS-300 / CVGS-500 ablation | UCI Covertype | `scripts/run_ablation_study_forest.py` | `results_forest/<run>/ablation/cvgs_*` |
| E7 | Vision NAS | sklearn digits | `run_vision_nas.py` → `scripts/run_vision_nas.py` | Console |
| E8 | Speed-up benchmark | Synthetic 50k × 25 | (reported in paper; no script in repo) | Paper Table I |
| E9 | Breast-cancer CASH | Wisconsin Breast Cancer | `XGBoostEvaluator` (no dedicated script) | Paper / comparative analysis |

The Covertype experiments (E4–E6) for the 19 November run were orchestrated by `run_experiments.ps1` (§8). A snapshot of the exact scripts used is kept in `scripts/Test1_19_Nov/`.

---

## E1 — Pathological demo (`main.py`)

- **Goal:** show model switching (XGBoost / LightGBM / RF) and the safeguards on a landscape built to make optimizers stagnate.
- **Setup:** `get_xgboost_search_space()` (9-D) with `PathologicalXGBoostEvaluator()` (3-fold CV).

| Mode | Budget | Pop | `t1_iters` | `k_iters` |
|---|---|---|---|---|
| Default (fast demo) | 120 | 20 | 3 | 2 |
| `--full` | 2500 | 100 | 10 | 5 |

You can also set `--budget`, `--pop_size`, `--t1_iters` and `--k_iters` directly.

## E2 — UPGS ablation (`run_ablation_study.py`)

- **Question:** can any single optimizer, running alone, match EvoSynth's coordinated portfolio? And what if each optimizer gets an equal 25% share of the budget?
- **Setups** (Uniform Production Grade Setups):

| Setup | Budget | Pop | 25% checkpoint | Evaluator mode |
|---|---|---|---|---|
| UPGS-120 | 120 | 50 | 30 evaluations | 3-fold CV ("defence mode") |
| UPGS-2500 | 2500 | 50 | 625 evaluations | `fast_mode` single 75/25 split ("production scale") |

- **Isolated runs.** GA, PSO, DE and CMA-ES each run alone under strict rules:
  - each starts from its **own** random population of `pop_size`;
  - there is no co-evolution, no elite sharing and no swapping;
  - each run gets the full budget;
  - each run uses the same space, evaluator and seed policy.

  Mechanics:
  - Generation 0 evaluates `pop_size` random vectors in chunks.
  - The run then loops `ask(batch)` → evaluate → `tell`, with `batch = max(5, min(10, pop//5))`, i.e. 10 for a population of 50.
  - **Passive telemetry** is logged at every step: diversity, trend, speed `1/(t+1e-4)`, efficiency `best/ln(max(1,t)+e)`, composite (0.30/0.25/0.15/0.15/0.15 with fixed scalings) and UCB (`c = 0.5`). It is **never used to make decisions**.
  - At the 25% checkpoint the state is captured: best accuracy, diversity and UCB.
- **EvoSynth run:** the same budget and population, with `t1_iters = max(3, int(budget·0.05 / (pop·0.4/4)))` and `k = max(2, t1_iters//2)`. The checkpoint accuracy is read from `optimizer.history`.
- **Options:**

| Flag | Values | Default |
|---|---|---|
| `--setup` | `120`, `2500`, `both` | `120` |
| `--dataset` | `pathological`, `breast_cancer`, `covtype` (10k subsample) | `pathological` |
| `--max_workers` | integer | `min(8, cpu)` |

- **Outputs:**
  - `results/individual_runs/<SETUP>_<ALGO>.txt`: iteration log, checkpoint audit, final audit, discovered hyperparameters;
  - `<SETUP>_EvoSynth.txt`;
  - `results/ablation_upgs_<n>_summary.txt` (Part 1: the 25% checkpoint; Part 2: full budget; plus a conclusion);
  - `..._run.txt` (the summary plus per-iteration logs);
  - `results/csv/ablation_upgs_<n>.csv` (every evaluated individual).

## E3 — Covertype demo with refit (`run_covertype.py`)

- **Goal:** the headline real-world benchmark, covering model families, feature engineering and generalisation.
- **Search:** the 9-D space with `CovertypeEvaluator(subsample=20000)`, i.e. 2-fold CV on 20k stratified rows.

| Mode | Budget | Pop | `t1_iters` | `k` |
|---|---|---|---|---|
| Default | 120 | 20 | 3 | 2 |
| `--full` | 500 | 40 | 5 | 3 |

  You can also set `--subsample`.
- **Step 6 refit (the generalisation test):**
  1. Reload the full dataset and take a stratified split of **60,000 train / 20,000 test** (`random_state=42`).
  2. Decode the winner and apply its FE strategy (fit on train, transform test).
  3. Build the winning model at full capacity:

| Family | Refit model |
|---|---|
| ExtraTrees / RandomForest | `n_estimators = max(n, 150)`, `max_depth = depth·3` if depth < 10 else unlimited, `max_features = colsample`, `n_jobs = −1` |
| LightGBM | `n_estimators = max(n, 250)` |
| XGBoost | `n_estimators = max(n, 200)`, `mlogloss` |

  4. Fit on 60k, then print the search CV accuracy, the test accuracy, the architecture and the FE strategy.

## E4 — Covertype scaled runs (`run_covertype_evosynth.py`)

- **Goal:** the budget-scaling curve (120 → 300 → 500) with crash-safe checkpointing.
- **Options:**

| Flag | Values | Default |
|---|---|---|
| `--budget` | 120, 300, 500, 1000 | 300 |
| `--pop_size` | integer | 30 |
| `--checkpoint_interval` | integer | 10 |
| `--run_name` | e.g. `Test1_19_Nov` | — |
| `--output_dir` | path | — |

- **Schedule:** `t1_iters = max(3, int(0.03·budget))` and `k = max(2, int(0.02·budget))`:
  - 300 evaluations → 9 / 6;
  - 500 evaluations → 15 / 10.
- **Fault tolerance:**
  - `evaluator.evaluate` is wrapped so that every 10 evaluations (and at the end) an **atomic** JSON checkpoint is written (temp file then `os.replace`) to `evosynth/checkpoints/checkpoint_<budget>.json`;
  - the checkpoint holds `budget`, `evals_used`, `best_cv_accuracy`, `best_config`, `elapsed_seconds`, `status` and `timestamp`;
  - console output is teed to `evosynth/run_<budget>.txt`.
- **Refit:** the same 60k/20k refit as E3. The final checkpoint adds the generalisation accuracy, the architecture and the FE strategy.
- **The 19 November run used:**
  - budget 300 with pop 30;
  - budget 500 with pop 30;
  - the 120 point is the E3 demo run (pop 20).

## E5 — Baselines (`run_baselines_forest.py`)

All baselines use the **same 60k/20k split** (`random_state=42`).

| Baseline | Search | Refit / test |
|---|---|---|
| Default ExtraTrees (untuned floor) | None | `ExtraTreesClassifier(n_estimators=100)`, sklearn defaults, fit on 60k, test on 20k |
| Random Search | 120 uniform random latent vectors scored with `CovertypeEvaluator(20k)` | Decode the best; refit **its model family and FE strategy** with `n_estimators=150` and library defaults (the tuned hyperparameters are **not** used) |
| Optuna TPE | 120 trials of `TPESampler(seed=42)` over 9 `x_i ∈ [0,1]` floats, scored with `CovertypeEvaluator(20k)`; SQLite storage `optuna_covertype.db` with `load_if_exists` (resumable) | Same simplified refit as Random Search |
| FLAML AutoML | `AutoML.fit` on the 60k training rows; `time_budget = 585 s` (about 10 min, matched to EvoSynth); `metric = accuracy`; estimators `lgbm`, `rf`, `extra_tree`, `xgboost`; holdout evaluation chosen by FLAML; seed 42 | FLAML's own final model, tested on 20k |

- **Options:**
  - `--test`: `all` (default), `extratrees`, `random`, `optuna` or `flaml`;
  - `--trials`: default 120;
  - `--time_budget`: default 585;
  - `--run_name` / `--output_dir`.
- **Outputs:** `default_extratrees.txt`, `random_search.txt`, `optuna.txt` (and `.db`), `flaml.txt` (and `flaml_run.log`).

**Default XGBoost / LightGBM / RandomForest (80.97 / 81.07 / 81.48%).** These floors appear in the paper, the slides and the master table. The repository has no script that produces them; they were obtained earlier and are reported as values.

## E6 — CVGS ablation on Covertype (`run_ablation_study_forest.py`)

- **Goal:** repeat the isolation ablation on real data.
- **Setups:**

| Setup | Budget | Pop | 25% checkpoint |
|---|---|---|---|
| CVGS-300 | 300 | 30 | 75 evaluations |
| CVGS-500 | 500 | 30 | 125 evaluations |

  The evaluator is `CovertypeEvaluator(20k)`.
- **Differences from E2:**
  - **Resumable.** For each optimizer, a JSON (metadata and logs) and a pickle (best individual, population, all individuals) are saved atomically every 10 evaluations under `ablation/<setup>/checkpoints/`. On restart, completed optimizers are skipped and interrupted ones resume from their last evaluation. `experiment_state.json` tracks the status of each algorithm.
  - Batch size is `max(5, min(10, pop//3))`.
  - EvoSynth uses `t1_iters = max(3, int(0.04·budget))` and `k = max(2, int(0.02·budget))`.
  - The orchestrator runs it with `--max_workers 2` to limit RAM.
- **Outputs:**
  - `individual_runs/CVGS-<n>_<ALGO>.txt`;
  - `csv/ablation_cvgs_<n>.csv`;
  - `ablation_cvgs_<n>_summary.txt`;
  - checkpoints.

## E7 — Vision NAS (`run_vision_nas.py`)

- **Goal:** show that the engine is not limited to tabular data.
- **Setup:** `get_image_nas_search_space()` (7-D), `ImageCNNEvaluator(epochs=2, batch_size=64)`, budget 80, pop 20, `t1_iters = 3`, `k = 2`.
- **Output:** the console log and the best CNN topology.

## E8 — Speed-up benchmark (paper Table I)

Compares three evaluation modes:
- serial (1 core);
- parallel batching (8 threads);
- parallel plus early stopping (8 threads).

It measures the initial 100-individual population and a full 120-evaluation run on a synthetic classification dataset (50,000 rows, 25 features, as described in the comparative analysis).

## E9 — Breast-cancer CASH

Uses an 8-dimensional space (`model_type` plus 7 hyperparameters, without FE) with `XGBoostEvaluator` (3-fold CV) and 120 evaluations on 8 threads. It reports the selected architecture and the discovered parameters.

---

## 8. `run_experiments.ps1`, the staged orchestrator

```
param([string]$Stage = "menu", [string]$RunName = "")
```

- **Run folder:** `results_forest\Test<N>_<dd_MMM>`. It reuses the latest folder for today's date or creates `Test1_<date>`, and makes the subfolders `baselines`, `evosynth`, `ablation\cvgs_300`, `ablation\cvgs_500` and `summary`.
- **Machine profile in the header:** Intel Core i5-12450HX (8 cores / 12 threads), 16 GB RAM.

| Stage | Command | Estimated duration |
|---|---|---|
| 1 Baselines | Installs optuna / flaml if missing, then `run_baselines_forest.py --test all --run_name $RunName` | ~15–20 min (Random Search and Optuna took ~35 min each in practice) |
| 2 EvoSynth scaled | `run_covertype_evosynth.py --budget 300 --pop_size 30 ...`, then `--budget 500 ...` | ~1 h |
| 3 CVGS-300 | `run_ablation_study_forest.py --setup CVGS-300 --pop_size 30 --max_workers 2 --checkpoint_interval 10` | Overnight |
| 4 CVGS-500 | The same with `--setup CVGS-500` | Overnight |
| A | Stages 1 → 4 in sequence, reporting total hours | — |
| N | Creates a new `Test<N+1>_<date>` folder and reopens the menu | — |

Usage: `.\run_experiments.ps1` (interactive menu), or `.\run_experiments.ps1 -Stage 2 -RunName Test1_19_Nov`.

The `summary/` folder holds `main_comparison_table.txt` and `convergence_curve.csv`. These were compiled from the stage outputs; no script in the repository generates them.
