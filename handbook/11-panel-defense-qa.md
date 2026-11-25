# 11 — Panel Defence & Viva Q&A

These answers are grounded in the code and results in files 03–08. Where the older defence material overstated something, the answer below gives the accurate version.

---

## 1. The 30-second pitch

> EvoSynth is a dynamic meta-optimizer for AutoML. Most tuners bet the whole budget on one search algorithm. EvoSynth instead runs four complementary optimizers (GA, PSO, DE and CMA-ES) in a two-tier island model. A UCB-style controller scores each one on fitness, improvement and diversity, promotes the strongest, swaps in a challenger when the incumbent stalls, and shares elite solutions between islands. Model family and feature engineering are genes in the same latent space, so EvoSynth chooses them too. The output is a single deployable model, not an ensemble. On UCI Forest Covertype (581k rows) it found ExtraTrees with automatically generated interaction features, which reached 94.05% held-out test accuracy at 500 evaluations. That beats untuned models (81–89.6%), Optuna (89.8%) and FLAML (92.4%), and every decision is visible live on a dashboard.

## 2. Key numbers to remember

| Item | Value |
|---|---|
| Dataset | Covertype: 581,012 rows, 54 features (10 continuous, 44 binary), 7 classes. Classes 1 and 2 make up more than 84%. |
| Search protocol | 20k stratified rows, 2-fold CV. Refit on 60k train, test on 20k unseen rows. |
| EvoSynth test accuracy | 93.47 / 93.82 / 94.05 at 120 / 300 / 500 evaluations |
| Baselines | Default XGB 80.97, LGBM 81.07, RF 81.48, ET 89.59; Optuna 89.80; FLAML 92.44; Random Search 93.92 |
| Error reduction (120 run, against an untuned RF/XGB-type floor) | about 19% → 6.53%, a 64.6% relative reduction |
| SWAP event | T1_PSO UCB 0.720 > T2_GA 0.547 + 0.10, so PSO was promoted, and T2_PSO produced the 120-run winner |
| Speed-up | Initial population 44.8 s → 3.9 s (11.5×); 120-evaluation run 392 s → 64 s |
| Constants | 40/60 split; c = 0.1; swap margin 0.10; grace 5 iterations; cooldown 2k; elite 20%; injection 50% (T1) / 20% (T2); stagnation limit 3 |
| Composite weights | Fitness 0.30, speed 0.25, efficiency 0.15, diversity 0.15, trend 0.15 |
| NAS | 95.33% (paper) / 95.78% (deck) on digits in 80 evaluations |

---

## 3. Questions and answers

### A. Novelty and motivation

**Q1. What is novel? Isn't this just another hyperparameter tuner?**
Optuna, Hyperopt and FLAML each commit to a single search strategy for the whole budget. By No Free Lunch, no single strategy suits every landscape. EvoSynth is an adaptive meta-optimizer:
- it runs a portfolio of four population-based optimizers in two tiers;
- it scores them online on fitness, improvement trend and diversity, plus an exploration bonus;
- it reallocates budget by promotion, live SWAP with rollback, stagnation injection and elite migration.

The model family and the feature-engineering strategy are searched in the same loop. The meta-decisions are observable live.

**Q2. Why not just ensemble, as AutoGluon or H2O do?**
Ensembles act at prediction time: every query runs 10–50 models, which multiplies latency and memory. EvoSynth uses multiple *search algorithms* only offline and ships *one* model, so serving costs no more than for that single model.

**Q3. How does EvoSynth deal with the No Free Lunch theorem?**
It does not try to beat NFL with a better single algorithm. It hedges across complementary algorithms:
- GA explores;
- PSO exploits quickly;
- DE explores robustly;
- CMA-ES fine-tunes.

It then shifts budget towards whichever one is currently making progress on *this* problem.

**Q4. How is EvoSynth different from algorithm portfolios or hyper-heuristics?**
It combines four things: a population-level portfolio, a two-tier structure that separates exploration (Tier 1) from exploitation (Tier 2), a composite reward that includes diversity and trend, and the transfer of actual solutions between algorithm families through elite migration and swap seeding. It also searches CASH and feature engineering jointly.

### B. Dataset and validation

**Q5. Why switch from the synthetic dataset to Covertype?**
On the 30% label-noise synthetic set, every method scored in a narrow 79–82% band. The improvements were tenths of a point, too small to demonstrate value. Covertype is real, published and imbalanced, with non-linear structure. There the search space matters: model family and feature engineering move test accuracy from about 81% to 94%.

**Q6. You searched on only 20k of 581k rows. How can you claim results on Covertype?**
This is multi-fidelity evaluation, the standard trade-off also used by FLAML and Auto-sklearn:
- 120–500 cross-validated evaluations on all 581k rows would take many hours on a laptop, so the search ranks candidates on a stratified 20k slice;
- only the single winner is then refit on 60k rows and tested on 20k *unseen* rows;
- every baseline is reported on the same 60k/20k split.

**Q7. Is there data leakage?**
No:
- the subsample and the splits are stratified with a fixed seed;
- the feature engineer is `fit` on training folds and only `transform`s validation and test data;
- the 20k test rows are never seen during search.

**Q8. Why is test accuracy (94%) so much higher than search CV (86.6%)?**
During search each fold trains on about 10k rows and RF/ET are capped at 150 trees. The refit trains on 60k rows with at least 150 trees. Tree ensembles on Covertype gain a lot from more data. The gap reflects training-set size, not leakage, and the same pattern appears for Random Search (86.33 → 93.92).

**Q9. Why is 81% accuracy misleading on Covertype?**
Two classes make up more than 84% of the rows, so a model can score about 81% while doing poorly on the minority classes (Aspen, Cottonwood/Willow, Douglas-fir). Moving to 94% means correctly classifying many of those harder samples.

### C. Mechanics and maths

**Q10. Explain the composite score. Why not reward accuracy alone?**
An island can hold a high best score while its population has collapsed into one basin and stopped improving. A pure-accuracy bandit would keep feeding it. The composite score min–max normalises across islands:

`0.30·fitness + 0.25·speed + 0.15·efficiency + 0.15·diversity + 0.15·trend`

- Speed and trend are both "improvement since the last scoring call", so they reward momentum.
- Diversity is `std/spread` of the island's fitness values.
- Efficiency is currently a constant placeholder.

The real Step 4 table shows the effect: CMA-ES was promoted with the lowest accuracy because it was the only optimizer still improving.

**Q11. What is the UCB term doing?**
`UCB = composite + 0.1·sqrt(ln N_total / n_i)`. This is the UCB1 exploration bonus. It grows for optimizers that have consumed fewer evaluations, so an optimizer that was starved early still gets a chance. With c = 0.1 it mostly breaks ties (about 0.03–0.08).

**Q12. What triggers a SWAP, and what happened in your run?**
At each checkpoint (every k iterations), let `best_T1` be the Tier 1 optimizer with the highest UCB score and `worst_T2` the Tier 2 optimizer with the lowest. A swap happens when:
- `UCB(best_T1) > UCB(worst_T2) + 0.10`;
- `best_T1` is not cooling down;
- no grace period is active.

The swap itself:
1. The challenger takes over the Tier 2 island, seeded with its own island plus the tail of the old Tier 2 population.
2. The demoted optimizer gets the top Tier 2 individuals in Tier 1 and a 2k cooldown.
3. A 5-iteration grace period starts. If the global best does not improve within it, the swap is rolled back from a snapshot.

In the 120-evaluation Covertype run, T1_PSO (0.720) beat T2_GA (0.547) and was promoted. T2_PSO then produced the final best configuration.

**Q13. What is the universal latent space, and why is it needed?**
PSO velocities, DE differences, CMA-ES covariance and GA blending all need continuous vectors. Every hyperparameter is mapped to `[0,1]`:
- continuous values are scaled linearly or logarithmically;
- integers are rounded;
- categoricals with K options use `min(⌊zK⌋, K−1)` bins.

So every optimizer can move through "XGBoost vs ExtraTrees" or "none vs interactions" as smoothly as through a learning rate.

**Q14. How does the feature engineering stay dataset-agnostic?**
`GenericFeatureEngineer` uses no column names:
- it treats columns with more than 15 distinct values as continuous;
- `stats` adds the row mean, standard deviation, minimum and maximum;
- `interactions` also adds pairwise differences and Euclidean norms for the six highest-variance continuous columns (54 → 88 features on Covertype).

The strategy is a searchable gene, and the search consistently picked `interactions`.

**Q15. How do you prevent ping-ponging and bad promotions?**
- **Cooldown:** the demoted optimizer cannot come back for 2k iterations.
- **Grace period and rollback:** a promoted optimizer must improve the global best within 5 iterations.
- **Margin:** swaps need a lead of more than 0.10.
- **Single active grace period:** only one swap can be pending at a time.

**Q16. What happens to stagnating islands?**
- In Tier 1, an island with zero trend at a checkpoint has 50% of its members replaced by fresh random vectors.
- In Tier 2, after 3 consecutive zero-trend checkpoints, the bottom 20% are replaced.

Elite sharing between the two Tier 2 islands carries good solutions across algorithm families. Tier 1 is deliberately excluded so it stays a diverse sandbox.

**Q17. Can a demoted optimizer return? Does the population size change?**
- **Return:** yes. All four optimizers stay alive, and once its cooldown ends, a Tier 1 optimizer can be swapped back in.
- **Population size:**
  - total *evaluations* are fixed by the budget;
  - island sizes are fixed per tier (Tier 2 islands are about 4× Tier 1 islands);
  - injection replaces members rather than adding them.

### D. Performance and trade-offs

**Q18. Doesn't running four algorithms cost four times as much?**
No. The budget is a fixed number of evaluations shared by all optimizers: 40% of the initial population seeds Tier 1, and the rest of the budget is spent by whichever optimizers are active. Scoring, sorting and swapping cost microseconds, and more than 99.9% of the time goes on model training, as in any tuner.

**Q19. How did you speed it up?**
- Batches are evaluated in a `ThreadPoolExecutor` (up to 8 threads). XGBoost, LightGBM and sklearn release the GIL during training.
- XGBoost and LightGBM stop after 10 non-improving rounds.
- RF and ET are capped at 150 trees during search.

The paper reports initial-population evaluation going from 44.8 s to 3.9 s (11.5×).

**Q20. Default models score 81% in 2 seconds. Why spend 10–40 minutes?**
Because the gain is large: up to 94% test accuracy, about a two-thirds cut in errors compared with that kind of floor, and much better minority-class coverage. The cost is paid once, offline. Serving costs the same as one model.

### E. The hard questions (answer honestly)

**Q21. Random Search reached 93.92% with 120 evaluations. Isn't EvoSynth unnecessary?**
On this space, the good region (ExtraTrees with interactions) is broad and easy to hit, and every method's search CV saturates at about 86.3–86.6%. EvoSynth at 500 evaluations still has the best test result (94.05%). Also:
- Random Search's refit used library defaults with 150 trees, not tuned hyperparameters, so the protocols differ;
- EvoSynth's contribution is robustness across landscapes, and its value grows where a single strategy stalls.

The honest conclusion is that Covertype did not strongly separate the search strategies, while feature engineering and model choice mattered most.

**Q22. In the ablations, an isolated optimizer sometimes beat EvoSynth. Why?**
The results on Covertype:
- CVGS-300: −0.06 points against the best isolated optimizer;
- CVGS-500: about equal.

You only know *which* single optimizer wins after the run, and it differs by landscape and budget. EvoSynth tracks the best of the four without that hindsight. The margins are small on these flat landscapes, and the results come from single runs.

**Q23. The ablation tables show EvoSynth diversity 0.500 / 0.450. Are those measured?**
No. In the ablation code, EvoSynth's diversity, composite and UCB entries are fixed constants. Only its accuracy, winner, hyperparameters and runtime are measured. Inside real runs, diversity is std/spread of fitness, and with two-member islands it is always exactly 0.5 (or 0). So diversity-based claims should rely on the live logs, not on those ablation columns.

**Q24. Is the UCB1Bandit class actually used?**
The promotion and swap decisions use the controller's inline formula, composite plus `0.1·sqrt(ln N / n_i)`. The `UCB1Bandit` class records rewards at promotion, but its `select_next` method is not called. The "bandit" behaviour is the UCB-scored promotion and swapping.

**Q25. What are the limitations?**
- It handles tabular data and small CNNs, with no NLP or audio.
- It runs on a single node (threads, no Ray).
- There is no successive halving.
- Boosting-only genes are inactive for RF and ET.
- The efficiency metric is a placeholder.
- Results come from single runs.
- Choosing the maximum of noisy 2-fold CV may overtune.

**Future work:**
- noise-aware or "confirm before select" selection;
- cost-aware UCB;
- surrogate pre-screening;
- distributed workers;
- warm starts across datasets.

**Q26. Why SSE rather than WebSockets? What does the dashboard show?**
Telemetry flows one way, and SSE is simpler, reconnects automatically and passes through firewalls easily. The dashboard shows:
- the current workflow step and budget progress;
- which optimizers are in Tier 1 and Tier 2;
- each family's best accuracy;
- a feed of every promotion, swap, rollback, injection and elite share;
- the full log, including the UCB scoring tables.
