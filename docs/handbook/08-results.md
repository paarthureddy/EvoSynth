# 08 — Results

Every figure below is taken from files in the repository: the panel-review (v1) results archived in `results/archive/2026-11-19_covertype_v1/` and the paper in `paper/`. They were produced by the v1 code (git tag `v1-panel-review`) and will be replaced by the re-runs after the Phase 0–5 fixes. Each section names its source. Accuracy is in %. "CV" means search-phase cross-validation accuracy. "Test" means accuracy on the 20k held-out Covertype rows.

---

## 1. Covertype master comparison

**Protocol:**
- Every method is tested on the same stratified **60,000 train / 20,000 test** split (`random_state = 42`).
- Searches score candidates on a 20,000-row stratified slice with 2-fold CV.
- **Source:** `results/archive/2026-11-19_covertype_v1/summary/main_comparison_table.txt` and the baseline and run files.

| Category | Method | Search budget | CV (20k) | **Test (60k/20k)** | FE | Winning model |
|---|---|---|---|---|---|---|
| Untuned floor | Default XGBoost | 0 | — | 80.97 | none | XGBoost |
| Untuned floor | Default LightGBM | 0 | — | 81.07 | none | LightGBM |
| Untuned floor | Default RandomForest | 0 | — | 81.48 | none | RandomForest |
| Untuned floor | Default ExtraTrees (100 trees) | 0 | — | **89.59** (fit 1.3 s) | none | ExtraTrees |
| Competitor | Optuna TPE | 120 trials (35.6 min) | 84.61 | **89.80** | interactions | XGBoost |
| Competitor | FLAML AutoML | 585 s (9.9 min) | holdout | **92.44** | internal | ExtraTrees (237 trees, max_features 0.563, max_leaves 14617, entropy) |
| Competitor | Random Search | 120 evaluations (34.0 min) | 86.33 | **93.92** | interactions | ExtraTrees |
| **EvoSynth** | @120 evaluations (pop 20) | 9.7 min | 86.39 | **93.47** | interactions | ExtraTrees |
| **EvoSynth** | @300 evaluations (pop 30) | 31.6 min | 86.39 (86.395) | **93.82** | interactions | ExtraTrees |
| **EvoSynth** | @500 evaluations (pop 30) | 37.3 min | 86.59 (86.595) | **94.05** | interactions | ExtraTrees |

**Budget-scaling curve** (`summary/convergence_curve.csv`):

| eval_budget | cv_accuracy | test_accuracy | search_time_min | architecture | fe |
|---|---|---|---|---|---|
| 120 | 86.39 | 93.47 | 9.7 | ExtraTrees | interactions |
| 300 | 86.39 | 93.82 | 31.6 | ExtraTrees | interactions |
| 500 | 86.59 | 94.05 | 37.3 | ExtraTrees | interactions |

**Winning configurations**, taken from the logs and checkpoints. Columns are in the latent space's alphabetical order.

| Run | Winner | booster | colsample | fe | lr | depth | model | n_est | reg_alpha | subsample |
|---|---|---|---|---|---|---|---|---|---|---|
| @120 | T2_PSO | gbtree | 0.4317 | interactions | 0.0148 | 8 | ExtraTrees | 334 | 5.7361 | 0.6725 |
| @300 | T2_CMAES | dart | 0.6185 | interactions | 0.0860 | 9 | ExtraTrees | 279 | 4.6919 | 0.7522 |
| @500 | T2_CMAES | gbtree | 0.3967 | interactions | 0.0109 | 10 | ExtraTrees | 182 | 9.9630 | 0.9998 |

For ExtraTrees only three of these matter: the number of trees, depth (×3, or unlimited when depth = 10) and colsample (used as `max_features`). The boosting-only columns (booster, lr, reg_alpha, subsample) are inactive.

**Claims the project makes from this table:**
- EvoSynth @500 is +13.08 points above default XGBoost and +4.46 above default ExtraTrees.
- It is +4.25 above Optuna TPE and +1.61 above FLAML.
- FLAML and EvoSynth independently chose ExtraTrees. EvoSynth's interaction features account for its extra margin over FLAML.
- Accuracy rises steadily with budget: 93.47 → 93.82 → 94.05.

**Points to keep in mind when reading this table:**
- **Random Search is a strong competitor.** At 120 evaluations it reached 93.92% test accuracy. That is above EvoSynth @120 (93.47) and @300 (93.82), and below EvoSynth @500 (94.05). On this space, search CV saturates at about 86.3–86.6% for every method, because the `ExtraTrees` + `interactions` region is easy to find.
- **The refit protocols differ** (see [07-experiments.md](07-experiments.md) E3):
  - Random Search and Optuna refit only their chosen *model family and FE strategy*, with 150 trees and library defaults; their tuned hyperparameters are discarded.
  - EvoSynth refits with its tuned hyperparameters.
  - FLAML trains its own final model.
  - Default ExtraTrees uses 100 trees.
- **The default XGBoost, LightGBM and RandomForest floors** (80.97 / 81.07 / 81.48) are reported values with no generating script in the repository. Default ExtraTrees (89.59) is a much stronger floor and is the fairer untuned reference.
- **Single run per configuration.** No repeated seeds and no confidence intervals were recorded.
- **Why test accuracy exceeds CV accuracy.** The search trains on roughly 10k rows per fold, with RF/ET capped at 150 trees. The refit trains on 60k rows with at least 150 trees. Tree ensembles on Covertype improve substantially with more data, so the gap does not indicate leakage. FE is fit on training data only.

---

## 2. Headline error-reduction claim (paper and slides, 120-evaluation run)

| Stage | Model | FE | Time | Accuracy | Error |
|---|---|---|---|---|---|
| Default baseline 1 | XGBoost | raw 54 | 2.1 s | 80.97 | 19.03 |
| Default baseline 2 | LightGBM | raw 54 | 1.2 s | 81.07 | 18.93 |
| Default baseline 3 | RandomForest | raw 54 | 3.4 s | 81.48 | 18.52 |
| EvoSynth, raw features only | XGBoost (DART, depth 10) | none | 6 min 14 s | 82.73 | 17.27 |
| EvoSynth search CV | ExtraTrees (334 trees, depth 8) | interactions | 8 min 30 s | 86.39 | 13.61 |
| **EvoSynth generalisation** | ExtraTrees, refit 60k / test 20k | interactions | 1 min 14 s | **93.47** | **6.53** |

- **Relative error reduction:** the project quotes 64.6% (19.0% → 6.53% error). Against the best default floor (RF, 18.52%) it is (18.52 − 6.53) / 18.52 ≈ 64.7%. Against default ExtraTrees (10.41% error) the reduction is about 37%.
- **Total time:** 9 min 44 s for the whole run (search plus refit).
- The "raw features" row (82.73%) comes from an earlier run without the FE gene and has no log in the repository.

---

## 3. CVGS ablation on Covertype (search CV only)

**Source:** `results/archive/2026-11-19_covertype_v1/ablation/`. Population 30, 2-fold CV on 20k rows.

### CVGS-300 (checkpoint at 75 evaluations)

| Strategy | @75 best | @75 div | @75 UCB | **@300 best** | @300 div | @300 UCB | Model (all with interactions) | Runtime |
|---|---|---|---|---|---|---|---|---|
| Isolated GA | 86.48 | 0.300 | 0.941 | 86.53 | 0.281 | 0.719 | ExtraTrees (197 trees, depth 10) | 37.0 min |
| Isolated PSO | 86.53 | 0.278 | 0.938 | **86.59** | 0.300 | 0.722 | ExtraTrees (207, 10) | 32.7 min |
| Isolated DE | 86.36 | 0.296 | 0.940 | **86.59** | 0.475 | 0.749 | ExtraTrees (500, 10) | 30.2 min |
| Isolated CMA-ES | 86.34 | 0.228 | 0.929 | **86.59** | 0.297 | 0.722 | ExtraTrees (293, 9) | 25.4 min |
| **EvoSynth** | 86.30 | 0.500† | 0.704† | 86.53 | 0.450† | 0.720† | ExtraTrees (347, 10) | 36.7 min |

- † **EvoSynth's diversity, composite and UCB values in the ablation are hard-coded constants** (checkpoint 0.500 / 0.625 / 0.704; final 0.450 / 0.654 / 0.720). Only EvoSynth's accuracy, runtime, winner and hyperparameters are measured, so the diversity comparison with the isolated runs is not a measured result. The isolated optimizers' values come from passive telemetry, which uses different formulas and a different `c` from the controller. Measuring them is Phase 4.1.
- **EvoSynth run details:** `t1_iters = 12`, `k = 6`.
  - Tier 1 injections hit T1_GA and T1_DE.
  - GA and CMAES were promoted.
  - The winner was `T2_CMAES`.
- **Conclusion printed:** EvoSynth −0.06 points versus the best isolated optimizer. All optimizers find ExtraTrees with interactions.

### CVGS-500 (checkpoint at 125 evaluations)

| Strategy | @125 best | @125 div | @125 UCB | **@500 best** | Model |
|---|---|---|---|---|---|
| Isolated GA | 86.50 | 0.300 | 0.941 | 86.53 | ExtraTrees + interactions |
| Isolated PSO | 86.41 | 0.278 | 0.938 | 86.41 | ExtraTrees + interactions |
| Isolated DE | 86.51 | 0.296 | 0.940 | 86.60 (checkpoint JSON: 86.59) | ExtraTrees + interactions |
| Isolated CMA-ES | 86.60 | 0.228 | 0.929 | 86.60 (checkpoint JSON: 86.59) | ExtraTrees + interactions |
| EvoSynth (CV) | 86.48 | 0.500† | 0.704† | 86.59 | ExtraTrees + interactions |
| EvoSynth (60k/20k refit test) | — | — | — | **94.05** | ExtraTrees + interactions |

- **The summary's conclusion:**
  - on the 20k slice every optimizer converges to about 86.6% (the architecture dominates);
  - after refit, EvoSynth's configuration reaches 94.05%, against 89.59% for untuned ExtraTrees and 89.80% for Optuna.
- **Provenance:**
  - `experiment_state.json` records all four isolated runs as completed. It records the EvoSynth step inside this ablation harness as still `"running"`.
  - The EvoSynth row's values (86.59 CV, 94.05 test) match the standalone E2 run at 500 evaluations.
  - The diversity and UCB columns reported at 500 evaluations repeat the CVGS-300 values.

---

## 4. The documented live SWAP (Covertype, 120 evaluations)

**Source:** `results/archive/2026-11-19_covertype_v1/evosynth/run_120.txt`.
**Setup:** pop 20, `t1_iters = 3`, `k = 2`.

```
STEP 1  Evaluated 20/20 → best 85.96%           (20/120)
STEP 2  Tier 1: 8 individuals (2 per optimizer) | Tier 2 reserve: 12
STEP 3  Iter  GA      PSO     DE      CMAES   Global
         1   85.96   83.98   83.20   81.97   85.96   28/120
         2   86.21   83.98   83.20   81.97   86.21   36/120  NEW BEST
         3   86.21   83.98   83.20   82.66   86.21   44/120
STEP 4  T1_CMAES 82.66 div .500 comp .625 bonus .079 UCB .704  → PROMOTED
        T1_GA    86.21 div .500 comp .525 bonus .079 UCB .604  → PROMOTED
        T1_DE    83.20 div .500 comp .271 bonus .079 UCB .350
        T1_PSO   83.98 div .000 comp .187 bonus .079 UCB .266
STEP 5  Iter T2_CMAES T2_GA  T1_DE  T1_PSO Global
         1   84.71   86.21  83.20  85.20  86.21   64/120
         2   84.71   86.33  83.20  85.20  86.33   84/120  NEW BEST
        CHECKPOINT: T1_PSO UCB .720 | T2_CMAES .665 | T2_GA .547 | T1_DE .292
        [SWAP] T1_PSO (0.720) beats T2_GA (0.547) by >10%   → PSO ↑ Tier 2, GA ↓ Tier 1
        [ELITE SHARE] top 20% pooled between T2 islands
         3   T2_CMAES 86.30  T2_PSO 86.30  T1_DE 83.20  T1_GA 86.33   104/120
FINAL   120 evals, 9m 44.4s, winner T2_PSO, 86.39% CV, ExtraTrees + interactions
REFIT   93.47% on 20k unseen test rows
```

**Interpretation:**
- **CMA-ES was promoted with the lowest accuracy.** It was the only optimizer still improving, and speed plus trend carry 40% of the composite weight.
- **PSO overtook GA by a wide margin.** PSO had climbed from 83.98 to 85.20 in Tier 1, while GA in Tier 2 was barely improving.
- **The promoted PSO produced the winning configuration** within the last partial iteration of the budget.
- The 0.500 diversities are a structural effect of two-member islands (see [03](03-engine-internals.md) §3).

**In the scaled runs** (@300: `t1_iters = 9`, `k = 6`; @500: `t1_iters = 15`, `k = 10`), the Tier 2 phase ended before it reached its first checkpoint. **No SWAP, injection or elite sharing happened in Tier 2** in those runs. Promotions were DE + CMAES (@300) and GA + CMAES (@500). The winner both times was `T2_CMAES`.

---

## 5. Results reported in the paper and docs without logs in the repository

| Experiment | Result | Source |
|---|---|---|
| **Speed-up** (100-individual initial population / 120-evaluation run) | Serial on 1 core: 44.8 s / 392.4 s. Parallel on 8 threads: 5.1 s / 78.2 s. **Parallel + early stopping on 8 threads: 3.9 s / 63.9 s.** This is the **11.5×** initial-population speed-up. | Paper Table I |
| **Vision NAS** (digits) | 80 evaluations: 95.33% in 14.6 s (paper), or 95.78% in 17.7 s (panel deck). Best topology (paper): conv1 = 16, conv2 = 64, fc = 64, dropout 0.0107, LeakyReLU, Adam with lr 0.0309. | Paper §III, `paper/slides.tex` slide 7 |
| **Inference footprint** | About 2 ms per prediction and about 150 MB RAM for the single winning model. Ensembles are quoted at 50–200 ms and 2–8 GB. | README, slides, comparative analysis (these figures are asserted; no measurement code is in the repo) |

---

## 6. Overall reading of the evidence

1. **The mechanics work as designed.**
   - The logs show promotion by composite score plus UCB, a live SWAP that promoted the eventual winner, T1 injections, elite sharing and rollback-ready grace periods.
   - The dashboard shows all of these as they happen.
2. **Accuracy against the field.**
   - On Covertype, EvoSynth's refit test accuracy (94.05% at 500 evaluations) is the best in the table.
   - It beats Optuna TPE, FLAML and the untuned floors.
   - Random Search comes close (93.92% at 120 evaluations), under its different refit protocol.
3. **Accuracy against isolated optimizers.**
   - In the Covertype ablations, EvoSynth's *search* accuracy is within 0.1 points of the best isolated optimizer.
   - It loses CVGS-300 by 0.06 and is within 0.01 on CVGS-500 (where its own ablation run did not finish).
   - Its practical value lies in not needing to know in advance which optimizer suits the landscape, and in its safeguards against collapse. It does not consistently beat the best single optimizer chosen after the fact.
4. **Feature engineering made the largest difference.**
   - The searchable `interactions` gene combined with ExtraTrees was the single biggest accuracy lever on Covertype.
   - Every search method that had access to that gene found it.
