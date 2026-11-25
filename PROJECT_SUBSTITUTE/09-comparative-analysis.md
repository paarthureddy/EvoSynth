# 09 — Comparative Analysis

This file positions EvoSynth against existing AutoML/HPO tools and against ensembling, and lists the related research. It condenses `docs/comparative_analysis.md` and the README and slide comparisons, and adds the measured Covertype comparison.

---

## 1. Taxonomy of approaches

```
                         ML optimisation paradigms
   ┌──────────────────┬──────────────────────┬──────────────────────┬────────────────────────┐
 Cost-frugal local   Bayesian + pruning     Post-hoc ensembling    Dynamic meta-optimisation
 search (FLAML)      (Optuna, SMAC)         (AutoGluon, H2O)       (EvoSynth)
 • greedy steps      • TPE/GP surrogates    • 10–50 models stacked • 4 competing EAs/swarms
 • cheap-first       • trial pruning        • high latency, RAM    • UCB-style meta-control
 • can trap on noise • one sampler/study    • O(M) inference       • island co-evolution
                                                                     • one model out (O(1))
```

---

## 2. Existing tools, and how EvoSynth differs

### Microsoft FLAML

- **Approach:** Cost-Frugal Optimization (CFO) and BlendSearch. It starts from cheap configurations (few trees, shallow, subsampled data) and climbs greedily along low-cost directions. It also offers zero-shot initialisation.
- **Strengths:** very fast on smooth tabular problems.
- **Limitations claimed by the project:**
  - it assumes a fairly smooth landscape, so greedy steps can stall in the nearest basin on noisy or multi-modal problems;
  - it runs a single search topology with no concurrent heuristics and no sharing between populations.
- **Measured on Covertype:** 92.44% test accuracy in 585 s with ExtraTrees, against EvoSynth's 94.05% at 500 evaluations. Both independently chose ExtraTrees.

### Amazon AutoGluon

- **Approach:** stacks and ensembles default models in multiple layers rather than tuning hard.
- **Strengths:** state-of-the-art accuracy on Kaggle-style tabular benchmarks.
- **Limitations:**
  - final models are 3–4-layer stacks of up to about 50 models, so latency jumps from about 1 ms to 100–250 ms;
  - artifacts reach 1–5 GB;
  - SHAP-style explanation of the stack is expensive or impractical.

### H2O AutoML / Driverless AI

- **Approach:** runs a fixed portfolio (GLM, DRF, XRT, GBM, XGBoost, deep learning) with random grid and Bayesian search under a time limit, then builds "All Models" and "Best of Family" stacked ensembles.
- **Strengths:** distributed scaling and rich feature engineering (target encoding, interactions).
- **Limitations:**
  - no knowledge transfer between algorithm silos;
  - JVM overhead;
  - peak accuracy depends on the ensemble.

### Optuna

- **Approach:** define-by-run API; TPE or CMA-ES samplers; ASHA/Hyperband pruning.
- **Strengths:** ergonomic and good at pruning.
- **Limitations:**
  - a `Study` commits to **one sampler**, with no portfolio and no budget reallocation between algorithms;
  - trials are independent, with no co-evolving population.
- **Measured on Covertype:** 84.61% CV and 89.80% test with 120 TPE trials (chose XGBoost with interactions).

### Auto-sklearn 2.0 (SMAC3)

- **Approach:** Bayesian optimisation with random-forest surrogates, meta-learned warm starts from OpenML, and a post-hoc ensemble.
- **Strengths:** fast warm starts and search over the whole pipeline.
- **Limitations:**
  - surrogate cost grows with dimensionality and budget;
  - it targets sklearn pipelines and has no native PyTorch NAS;
  - its output is an ensemble.

### Random Search (baseline)

- **Approach:** samples blindly with no feedback.
- **Measured on Covertype:** strong, at 86.33% CV and 93.92% test with 120 evaluations. It is a reminder that the 9-D space has a broad, easy-to-find good region (ExtraTrees with interactions).

---

## 3. Ensembling vs. search-time meta-optimisation

**How ensembles work:**
- voting or averaging: `ŷ = (1/M) Σ f_m(x)`;
- bagging: bootstrap resamples;
- stacking: base-model predictions form `Z ∈ R^{N×M}`, and a meta-learner `g` combines them.

They work because, by Condorcet-style arguments, models with uncorrelated errors that each beat chance reduce variance when aggregated.

**Production costs of ensembles** (the project's argument):
1. **Latency scales as O(M).** Every query runs all M models, and one slow model stalls the pipeline. Systems with sub-5 ms service-level targets (ad bidding, fraud detection, IoT) cannot afford a 20-model stack.
2. **Memory and infrastructure.** Gigabytes of RAM per replica multiply hosting costs.
3. **Fragility.** Many libraries must be upgraded in lockstep.
4. **Explainability.** SHAP for one tree model costs about O(T·L). For a multi-layer stack of 30 models it becomes impractical, which hurts in regulated settings.

Illustrative pipeline used in the docs:

```
Ensemble:  x → XGB 1.8ms, LGBM 1.2ms, CatBoost 3.5ms, RF 4.1ms, NN 64ms → GLM 0.5ms  ≈ 75 ms, 3.2 GB
EvoSynth:  x → single optimal model (e.g. LightGBM 1.2 ms)                            ≈ 1.2 ms, 18 MB
```

| Dimension | Ensembles (AutoGluon, H2O) | EvoSynth |
|---|---|---|
| When the multiplicity is used | At inference time | Offline, during search |
| Mechanism | Aggregates M models' predictions | Coordinates 4 search algorithms |
| Deliverable | M models | 1 model (or 1 CNN) |
| Latency | 50–200 ms | 1–3 ms (about 2 ms is the headline claim) |
| Memory | 1–8 GB | 10–50 MB per the comparative doc; about 150 MB per README and slides |
| Search-space discovery | None (models trained independently) | Co-evolution explores non-convex landscapes |
| Edge deployment | Impractical | Straightforward |

**Pareto framing ("is the advantage time or accuracy?").** EvoSynth claims Pareto efficiency: accuracy close to an ensemble, at the compute and latency cost of one model. The positions on the accuracy-versus-cost plane, as the project describes them:
- default models: about 81% in seconds;
- single optimizers: about 86–89%;
- EvoSynth: about 93–94% in under 40 minutes, with one model;
- heavy ensembles: about 94–96%, but over hours and with 10–15 models.

---

## 4. Comparison matrix

| Capability | Ensembling | FLAML | AutoGluon | Optuna | H2O | Auto-sklearn 2.0 | **EvoSynth** |
|---|---|---|---|---|---|---|---|
| Search strategy | None (post-hoc) | CFO local search | Random/Bayes + stacking | TPE / CMA-ES | Random grid + stacking | BOHB/SMAC + meta-learning | **UCB-scored dynamic meta-optimizer** |
| Concurrent heuristics | — | 1 | 1 | 1 sampler per study | 1 | 1 surrogate | **4 (GA, PSO, DE, CMA-ES)** |
| Live strategy switching | — | No | No | No | No | No | **Yes: tier SWAPs with grace and rollback** |
| Diversity safeguard | — | No | No | No | No | No | **Diversity in score, plus stagnation injection** |
| Knowledge sharing across algorithms | — | No | No | No | No | No | **Top-20% elite migration between T2 islands** |
| Search-space representation | Native | Native | Native | Define-by-run | Native | Native | **Universal latent `[0,1]^d`** |
| Model switching (CASH) | Stacks all | Sequential | Multi-model | User categorical | Portfolio | Pipelines | **`model_type` gene inside one search** |
| Feature engineering | Manual | Limited heuristics | Built-in | Manual | Fixed preprocessing | Fixed preprocessors | **Searchable dataset-agnostic gene** |
| NAS | No | No | Limited | Custom objective | No | No | **Native PyTorch CNN search** |
| Stagnation handling | — | Blind restart | — | Pruning | Wait for budget | — | **Trend tracking + random latent injection** |
| Deliverable | 10–30 models | 1 model | 15–50 models | 1 model | 10–30 models | Ensemble | **1 model** |
| Inference latency | 50–200 ms | 1–2 ms | 50–250 ms | 1–2 ms | 50–200 ms | Ensemble | **About 1–3 ms** |
| Observability | Opaque | Terminal logs | Static leaderboard | Optuna dashboard (polling) | Static UI | Tables | **Live SSE glass-box dashboard** |

Note that EvoSynth's latency is simply that of its chosen model family. The advantage comes from not being an ensemble, not from anything EvoSynth does at inference time.

---

## 5. Strengths and limitations (the project's own assessment)

**Strengths:**
1. **Portfolio meta-optimisation.** It does not bet on one sampler (the No Free Lunch argument).
2. **Co-evolution with live SWAPs and elite migration** helps the search escape stagnation.
3. **Zero inference penalty.** The output is one model.
4. **The universal latent space** lets continuous optimizers handle categorical choices and model families.
5. **Dataset-agnostic feature engineering** uses row statistics and interactions with no column knowledge.
6. **Real-time observability** of every meta-decision.
7. **CPU-only throughput:** thread parallelism plus early stopping gives more than 10× speed-up without a GPU.

**Limitations:**
1. **Modalities.** It handles tabular data and small CNNs only. There is no native NLP or audio support.
2. **No multi-fidelity scheduler.** Beyond early stopping and the fixed 20k slice, there is no ASHA/Hyperband-style successive halving across data fractions.
3. **Single node.** It uses `ThreadPoolExecutor` on one machine, with no Ray or Kubernetes distribution.
4. **Inactive dimensions.** Boosting-only genes drift when RF/ET is selected.
5. **Evidence from the results** ([08](08-results.md)):
   - the margins over the best isolated optimizer are small or negative in some ablations;
   - Random Search is competitive on Covertype;
   - all results come from single runs without repeated seeds.
6. **Possible overtuning.** Picking the maximum of noisy 2-fold CV scores on a 20k subsample may favour configurations that look good only because of noise. This motivates the research directions in §6.

**Future scope** (from the slides): distributed workers through Ray, warm-start meta-learning across datasets, and fuller deep-learning pipeline NAS.

---

## 6. Related research (`docs/AutoML_Relevant_Papers.txt`)

This list was compiled from the AutoML 2025 and 2026 accepted-paper lists: 21 of 69 papers were judged relevant, 5 of them highly. The 2026 entries were judged from titles only.

**High relevance:**
1. **Overtuning in Hyperparameter Optimization** (AutoML 2025, Best Paper). HPO picks configurations that look best because of validation noise and then generalise worse. EvoSynth's bandit greedily exploits noisy 2-fold CV on a 20k subsample, so this is the recommended core of the next direction.
2. **A Large-Scale Study of Overtuning Mitigation Strategies** (2026). Gives baselines for a noise-aware bandit and for "confirm before select".
3. **Zero-Shot Bayesian Optimization with TabPFN** (2026). TabPFN could pre-screen each optimizer's offspring so that only promising ones are evaluated (surrogate-assisted EvoSynth).
4. **From Tables to Runtime: Predicting Algorithm Runtime Distributions.** Would make UCB cost-aware (reward per predicted second) and give a real efficiency metric to replace the constant.
5. **Search More or Confirm Better? Reliable Selection in Small-Budget LoRA.** Supports a confirmation stage before the winner is selected.

**Medium relevance**, with the idea each one suggests:

| Paper | Relation to EvoSynth |
|---|---|
| Hyperparameter Optimization via Interacting with Probabilistic Circuits | Alternative surrogate and interactive priors |
| Ordered Transfer HPO | Warm-start the UCB statistics and population from earlier runs |
| Ax | A strong baseline |
| Regularized Neural Ensemblers | Ensemble the final population |
| EG-ENAS | Comparable to the vision NAS |
| The Ranking Trick | Rank-based UCB rewards instead of raw accuracy |
| Exploring One Million ML Pipelines | Support for the search-space design |
| BBOmix | Tabular lookup benchmarks would allow many seeds |
| MLJAR AutoML | Overlaps the glass-box observability claim |
| Rashomon sets in AutoML | The final population is a Rashomon set worth analysing |
| Algorithm Selection with Zero Domain Knowledge via Text Embeddings | Choose the initial bandit prior |
| Predicting Hyperparameter Importance from Meta-Features | Shrink or reweight the 9-D space per dataset |
| Dynamic Priors in BO | Conceptually close to the adaptive bandit |
| Efficient ensemble inference | Keeps a zero-penalty claim if the population is ensembled |
| MOEns | Multi-objective post-hoc selection |
| Neural Pipeline Search Spaces | Formal framing of the universal latent hypercube |

**Foundational references used across the project:**
- Wolpert & Macready 1997 (No Free Lunch)
- Auer et al. 2002 (UCB1)
- Kennedy & Eberhart 1995 (PSO)
- Storn & Price 1997 (DE)
- Hansen & Ostermeier 2001 (CMA-ES)
- Chen & Guestrin 2016 (XGBoost)
- Ke et al. 2017 (LightGBM)
- Bergstra & Bengio 2012 (random search)
- Bergstra et al. 2011 (TPE)
- Hutter et al. 2011 (SMAC)
- Fialho et al. 2010 (bandit-based adaptive operator selection)
- Wang et al. 2021 (FLAML)
- LeDell & Poirier 2020 (H2O)
- Zoph & Le 2017, Real et al. 2019 (NAS)
- Blackard & Dean 1999 (Covertype)
