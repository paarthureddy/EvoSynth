# 07 — Experiments

Every experiment is one `evosynth` command. All of them share the same dataset splits, evaluator, refit and seeding, so their numbers can be compared directly.

| # | Experiment | Data | Command | Run folder |
|---|---|---|---|---|
| E1 | Search with refit | any tabular dataset | `evosynth run [--full] --dataset NAME` | `results/search/<dataset>/<preset>/seed_<k>/` |
| E2 | Budget scaling | any tabular dataset | `evosynth scale --budget 300 \| 500 \| 1000` | `results/scaling/<dataset>/budget_<n>/seed_<k>/` |
| E3 | Baselines | any tabular dataset | `evosynth baselines --test all \| defaults \| random \| optuna \| flaml` | `results/baselines/<dataset>/<baseline>/seed_<k>/` |
| E4 | Ablation (+ tier-split ablation) | any tabular dataset | `evosynth ablation --setup CVGS-500 [--splits stratified,random,elite]` | `results/ablation/<dataset>/cvgs_<n>/seed_<k>/` |
| E5 | Vision NAS | Fashion-MNIST | `evosynth nas` | `results/nas/fashion_mnist/default/seed_<k>/` |
| E6 | Speed-up | any tabular dataset | `evosynth speedup` | `results/speedup/<dataset>/default/seed_<k>/` |
| — | Summary tables | all completed runs | `evosynth summary` | `results/summary/` |

Every run folder contains `manifest.json`, `result.json`, `run.log` and `events.jsonl` (see [02-architecture.md](02-architecture.md) §4). `--seed N` sets the search seed; `--seeds 1,2,3` repeats a command once per seed. `--subsample N` (run, scale, ablation, baselines, speedup) searches on a stratified N-row subset of the dataset's search rows. The panel-review (19 November) outputs are archived unchanged in `results/archive/2026-11-19_covertype_v1/`; they were produced by the v1 code at git tag `v1-panel-review` and are not comparable with the new runs.

## Shared protocol

- **Splits** (`datasets.partition`, `run.data_seed = 42`): held-out **test rows** split off first; **search rows** for every method's cross-validation; **train rows** for every refit. Test rows are never seen by any search.
- **Search score:** `TabularEvaluator`, 2-fold stratified CV on the search rows, in the dataset's metric (accuracy multiclass, ROC AUC binary); boosted models early-stop on a slice of the training fold.
- **Selection:** confirm before select. The top 5 distinct configurations are re-scored with 5-fold CV and the best confirmed one is selected (`[selection]`). These fits are not counted in the evaluation budget.
- **Refit:** `refit_and_score`: the selected configuration retrained on the train rows (boosted models early-stop on 15% of them), scored once on the test rows, with inference latency and model size measured (`models/inference.py`) and the model saved as `model.pkl`.
- **Same treatment for every tuned method.** Random Search and Optuna use the same evaluator, selection and refit as EvoSynth; only the search strategy differs.

## E1 — Search with refit (`evosynth run`)

- **Presets:** `search_demo` (default: budget 200, pop 40, t1_iters 3, k 2) and `search_full` (`--full`: budget 500, pop 40, t1_iters 5, k 3; k is lowered automatically if Tier 2 would get fewer than two checkpoints).
- **Steps:** the full `DynamicOptimizer` run ([03-engine-internals.md](03-engine-internals.md)), then confirm-before-select and the refit.
- **`result.json`:** metric, best search CV score, selected (confirmed) CV score and configuration, the confirmation table, winning algorithm, failed folds, evaluations, search seconds, and `refit` (test score, train / test rows, inference cost). `--no-refit` stops after selection.

## E2 — Budget scaling (`evosynth scale`)

- **Question:** how do search score and test score change with the budget?
- **Budgets** (`[experiments.scaling]`): 300, 500 (default) and 1000, population 40. `t1_iters = max(3, int(0.03·budget))`, `k = max(2, int(0.02·budget))`, giving Tier 2 about 2, 5 and 13 iterations.
- **Checkpoint:** `checkpoint.json` every 10 evaluations (best score and configuration so far).
- **Result:** as E1, plus the inference cost.

## E3 — Baselines (`evosynth baselines`)

| Baseline (folder) | Search | Final model |
|---|---|---|
| Untuned defaults (`defaults`) | None | Library defaults of XGBoost, LightGBM, RandomForest and ExtraTrees (seeded), trained on the train rows, scored on the test rows; inference measured for each |
| Random Search (`random_search`) | `--trials` uniform random latent vectors, scored with the same evaluator | Confirm-before-select + the shared refit |
| Optuna TPE (`optuna_tpe`) | `--trials` trials of `TPESampler(seed)` over the same latent space; SQLite study `optuna.db`, resumable | Confirm-before-select + the shared refit |
| FLAML (`flaml`) | `AutoML.fit` on the train rows with `--time-budget` seconds, the dataset's metric, estimators lgbm, rf, extra_tree, xgboost | FLAML's own refit; scored on the test rows |

For a fair comparison, give Random Search and Optuna the same number of evaluations as the EvoSynth run they are compared with, and FLAML the same wall-clock time (the Phase 5 runner does both).

## E4 — Ablation (`evosynth ablation`)

- **Question:** can any single optimizer, running alone, match EvoSynth's coordinated portfolio? And how much does the tier split matter?
- **Setups** (`[experiments.ablation].setups`): CVGS-500 (budget 500) and CVGS-1000 (budget 1000), population 40; comparison mark at 25% of the budget. EvoSynth uses `t1_iters = max(3, int(0.03·budget))`, so Tier 2 gets about 5 / 13 iterations.
- **Isolated runs.** GA, PSO, DE and CMA-ES each run alone: own random population, no tiers, no sharing, no injection, the full budget, the same evaluator and seed policy. Each step asks for a batch of `max(5, min(10, pop//3))` candidates.
- **Passive scoring (Phase 4.2).** Composite scores are relative, so the four isolated runs are scored *together* at every 10% of the budget (and at the mark) with the controller's own `CompositeScoringEngine` and UCB1 bonus. Nothing is decided from these scores; they show what the bandit would have measured.
- **EvoSynth.** One run per tier split in `--splits` (default `stratified`; `stratified,random,elite` is the split ablation). Its diversity, composite and UCB values come from the controller's `scoring_history` (the highest-UCB island at the last scoring call; diversity averaged over its islands).
- **Resume.** Each isolated optimizer's full state (optimizer object with its particles, targets or distribution, generators, every evaluated individual) is pickled every 10 evaluations under `checkpoints/`; an interrupted run continues exactly. A completed EvoSynth run is stored as a whole.
- **Outputs:** `summary.txt` (both marks, decisions, a conclusion generated from the numbers), `individual_runs/<SETUP>_<method>.txt`, `isolated_population.csv`, `population[_<split>].csv`, and `result.json` (every method's scores, diversity, composite, UCB, curve at the marks, best configuration; EvoSynth's decision counts).

## E5 — Vision NAS (`evosynth nas`)

- **Goal:** show that the engine is not limited to tabular data.
- **Setup:** `get_image_nas_search_space()` (7-D), `ImageCNNEvaluator(epochs=2, batch_size=64)`, budget 160, pop 40, `t1_iters = 3`, `k = 2` (`[experiments.nas]`), on Fashion-MNIST: 10,000 training and 2,500 validation images from the official training set.
- **Output:** the best CNN topology and its validation accuracy in `result.json`.

## E6 — Speed-up (`evosynth speedup`)

- **Question:** how much do parallel workers and early stopping actually save? (The v1 paper's 11.5× came from a synthetic 50k × 25 dataset.)
- **Method:** the same `configurations` (40) random configurations are evaluated four times: sequential with and without early stopping, and with `workers` threads (0 = all cores) with and without early stopping. Wall time, speed-up over the sequential no-early-stopping reference and mean CV score are reported (early stopping changes scores slightly).

## Summary tables (`evosynth summary`)

Reads every completed run and writes `results/summary/`: `main_table` (held-out test score per dataset and method, mean ± std over seeds, latency and model size), `ablation_table` (score at the mark and at the full budget, and how often EvoSynth beat the best isolated optimizer), `nas_table`, and `convergence/` (best-so-far curves averaged over seeds). Nothing is typed in by hand.

---

## 8. `tools/run_experiments.py`, the experiment runner

```
py -3.14 tools/run_experiments.py smoke            # run a plan
py -3.14 tools/run_experiments.py core --dry-run   # list its jobs and which are already done
```

| Plan | Seeds | Main comparison datasets | Also |
|---|---|---|---|
| `smoke` (Phase 5 as run) | 42 | Covertype, Adult | CVGS-500 on Covertype, speed-up, NAS |
| `lean` | 42–44 | Covertype, HIGGS, Adult, Poker Hand | CVGS-500 with the split ablation, speed-up, NAS |
| `core` | 42–44 | all 7 tabular datasets | as `lean` |

The main comparison per dataset and seed is: EvoSynth (`--full`, 500 evaluations), Random Search (500), Optuna TPE (500), untuned defaults, and FLAML with the same wall time as that EvoSynth search. Every search uses `--subsample 10000`. A job whose run folder is already `completed` is skipped, so the runner can be stopped and restarted; progress goes to `results/runner.log`, and `evosynth summary` runs at the end.

**Cost.** One evaluation takes about 1–9 s with 8 threads at 20,000 rows (MiniBooNE slowest, Poker Hand fastest); `smoke` takes about 6 hours, `core` about 2–2.5 days. The full original plan (all datasets, 5 seeds, every budget and ablation) would take about 3 weeks on one laptop.

---

## Quiz

**1.** Name the six experiment commands and what each answers.
<details><summary>Answer</summary>run (one search + refit), scale (score vs budget), baselines (reference methods), ablation (each EA alone, tier splits), nas (non-tabular search), speedup (what parallelism and early stopping save).</details>

**2.** What four things do all tuned methods share so that their test scores are comparable?
<details><summary>Answer</summary>The same splits, the same evaluator (CV, early stopping, metric), the same confirm-before-select and the same refit.</details>

**3.** How should Random Search, Optuna and FLAML be budgeted against an EvoSynth run of 500 evaluations?
<details><summary>Answer</summary>Random Search and Optuna: 500 evaluations each; FLAML: the same wall-clock seconds as that EvoSynth search.</details>

**4.** In the ablation, why are the isolated runs scored together rather than separately?
<details><summary>Answer</summary>Composite scores are min-max normalised across islands; one island alone would always tie. Scoring the four together reproduces what the controller would measure.</details>

**5.** Where do EvoSynth's diversity, composite and UCB in the ablation come from now?
<details><summary>Answer</summary>From the controller's own `scoring_history` (no constants).</details>

**6.** What does `--splits stratified,random,elite` add to an ablation?
<details><summary>Answer</summary>Two extra EvoSynth runs with the random and elite tier splits: the split ablation.</details>

**7.** An ablation is interrupted half-way through PSO. What happens when you re-run the command?
<details><summary>Answer</summary>Completed parts are skipped and PSO resumes from its last checkpoint with its full state (particles, generators), producing the same results as an uninterrupted run.</details>

**8.** Which plan was run in Phase 5, and why not the full plan?
<details><summary>Answer</summary>The `smoke` plan (seed 42, Covertype and Adult, ablation, speed-up, NAS); the full plan would take about three weeks on one laptop.</details>

**9.** What does `evosynth summary` produce, and from what?
<details><summary>Answer</summary>Mean ± std tables (main, ablation, NAS) and convergence curves in results/summary/, from every completed run folder.</details>

**10.** Why does Random Search take about three times longer than EvoSynth for the same 500 evaluations?
<details><summary>Answer</summary>It evaluates one configuration at a time; EvoSynth evaluates batches in parallel threads.</details>
