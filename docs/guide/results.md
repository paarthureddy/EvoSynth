# Results

These are the results of the first full run of the current engine through the experiment harness (the `smoke` plan: seed 42, Covertype and Adult, an optimizer ablation, a speed-up benchmark and NAS). Reproduce them with:

```bash
py -3.14 tools/run_experiments.py smoke
```

::: warning One seed
Every number on this page comes from a single seed. Differences of a few hundredths of a point between the tuned methods are ties until they are repeated over several seeds.
:::

## Protocol

| Item | Setting |
|---|---|
| Datasets | **Covertype** (581,012 rows, 7 classes, accuracy) and **Adult** (48,842 rows, binary, ROC AUC) |
| Held-out test rows | Covertype 20,000, Adult 12,000. Split off first, never seen by any search or selection, scored once |
| Search | 10,000 rows per dataset, stratified 2-fold cross-validation |
| Selection | The 5 best distinct configurations are re-scored with 5-fold CV; the best confirmed one is chosen |
| Refit | The chosen configuration is trained once on the train rows (Covertype 60,000, Adult 36,000) |
| Methods | EvoSynth, Random Search and Optuna TPE with 500 evaluations each and the same selection and refit; FLAML with the same wall time as EvoSynth; each model family with library defaults |
| Splits | The exact row indices are published in [EvoSynth-Bench on Zenodo](https://doi.org/10.5281/zenodo.23645119) |

## Held-out test scores

| Method | Covertype accuracy | Search time | Adult ROC AUC | Search time |
|---|---|---|---|---|
| Untuned XGBoost | 85.58 | – | 92.70 | – |
| Untuned LightGBM | 84.69 | – | 92.92 | – |
| Untuned RandomForest | 89.09 | – | 90.74 | – |
| Untuned ExtraTrees | 89.77 | – | 89.15 | – |
| FLAML | 92.53 | 13.0 min | **93.13** | 3.0 min |
| Random Search | **93.89** | 42.4 min | 92.98 | 19.1 min |
| Optuna TPE | 93.87 | 49.8 min | 92.95 | 3.3 min |
| **EvoSynth** | **93.89** | **12.2 min** | 92.96 | **3.0 min** |

- **EvoSynth is level with Random Search and Optuna** (within 0.03 points on both datasets) and selects the right model family for each: ExtraTrees with interaction features on Covertype, XGBoost on Adult.
- **Model choice matters most.** On Covertype the tuned searches are 4.1 points above the best untuned model; on Adult, where default boosting is already strong, only 0.03–0.06.
- **FLAML** is 1.4 points lower on Covertype and the best on Adult.
- **EvoSynth's shorter search time** comes from evaluating candidates in parallel (the baselines ran one trial at a time).
- **Why the tuned methods tie on Covertype:** the best configurations all build the same effective model during the search (forests are capped at 150 trees, and the same depth and number of features follow from the genes), so every good search reaches the same score.

## Optimizer ablation (Covertype, 500 evaluations each)

Each optimizer was also run alone with the full budget; all five runs were scored by the same [composite scoring](../concepts/composite-scoring.md) engine.

| Method | Best search CV at 125 evaluations | at 500 evaluations | Diversity at 500 |
|---|---|---|---|
| GA alone | 81.53 | 82.21 | 0.085 |
| PSO alone | 82.35 | 82.35 | 0.093 |
| DE alone | 82.13 | 82.34 | 0.363 |
| CMA-ES alone | 81.89 | 82.35 | 0.227 |
| **EvoSynth** | 82.08 | **82.35** | 0.206 |

EvoSynth matches the best single optimizer at the full budget without knowing in advance which one that is. The best single optimizer changes from run to run, which is the situation the [meta-optimizer](../concepts/meta-optimizer.md) is designed for.

## The meta-optimizer at work

From the Covertype run's log:
1. After 5 Tier 1 iterations, **DE** and **CMA-ES** were promoted. DE had a lower best score than CMA-ES but the highest composite score, because it was still improving and had the most diverse island.
2. At the first checkpoint **GA** displaced DE by a **SWAP** (UCB 0.592 against 0.369).
3. GA did not improve the global best within its 5-iteration grace period, so the swap was **rolled back**.
4. **CMA-ES** found the winning configuration: 82.35% search CV, 84.84% after 5-fold confirmation, 93.89% on the held-out test rows.

On Adult, GA and PSO were promoted, two swaps followed, and PSO found the winner.

## Speed-up

The same 40 Covertype configurations evaluated under four conditions on a 12-thread CPU (`evosynth speedup`):

| Condition | Time | Speed-up |
|---|---|---|
| Sequential | 255.6 s | 1.00× |
| Sequential + early stopping | 219.0 s | 1.17× |
| Parallel (12 threads) | 43.7 s | 5.85× |
| **Parallel + early stopping** | **41.9 s** | **6.10×** |

## Inference cost of the selected model

EvoSynth delivers one model, so serving costs whatever that model costs (single row, one thread):

| Selected model | Latency per row | Size |
|---|---|---|
| Covertype: ExtraTrees + interaction features | 12.5 ms | 470 MB |
| Adult: XGBoost | 0.26 ms | 0.4 MB |

## Neural architecture search

On Fashion-MNIST (160 evaluations, each CNN trained for 2 epochs on 10,000 images), EvoSynth reached **81.8%** validation accuracy with no changes to the engine.

## Next

The `lean` plan repeats the comparison over 3 seeds on Covertype, HIGGS, Adult and Poker Hand, which can show whether any of the differences above are real.
