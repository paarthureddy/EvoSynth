# 11 — Panel Defence & Viva Q&A

This chapter is the single place to prepare a review or viva. It merges the earlier Q&A chapter and the separate "Master Panel Defense Guide" (which is retired), and every number in it is from the Phase 5 smoke run ([08](08-results.md)). Where older defence material overstated something, the answer here gives the accurate version: a panel that checks a claim and finds it false will distrust everything else.

---

## 1. The 30-second pitch

> EvoSynth is an AutoML meta-optimizer built on evolutionary algorithms. Most tuners bet the whole budget on one search algorithm chosen in advance, but no algorithm is best everywhere. EvoSynth runs four (GA, PSO, DE and CMA-ES) in a two-tier island model. A UCB1 bandit scores each one on fitness, improvement, efficiency and diversity, promotes the strongest two, swaps in a challenger when one stalls (and rolls the swap back if it does not pay off), and shares elite solutions between them. Model family and feature engineering are genes of the same search, so it also chooses the model. It confirms the best candidates with 5-fold CV and ships one model, scored once on held-out data. On Covertype and Adult it matches Random Search and Optuna under an identical protocol, picks the right model family for each, matches the best single optimizer without knowing which one that is, and is the fastest on Covertype; every decision is visible live on a dashboard.

## 2. Key numbers to remember

| Item | Value |
|---|---|
| Datasets | **Covertype**: 581,012 rows, 54 features (10 continuous, 44 binary), 7 classes, two classes > 84%; accuracy. **Adult**: 48,842 rows, 14 features, binary; ROC AUC. Eight datasets are registered |
| Protocol | Test rows split off first (20,000 / 12,000), never searched, scored once. Search on 10,000 rows with 2-fold CV. Top 5 confirmed with 5-fold CV. Refit on train rows (60,000 / 36,000). Splits on Zenodo (EvoSynth-Bench v2.0.0) |
| Budget | 500 evaluations, population 40, seed 42 |
| Covertype test accuracy | **EvoSynth 93.89**, Random Search 93.89, Optuna 93.87, FLAML 92.53, untuned 84.69–89.77 |
| Adult test ROC AUC | FLAML **93.13**, Random Search 92.98, **EvoSynth 92.96**, Optuna 92.95, untuned 89.15–92.92 |
| Search time (Covertype / Adult) | EvoSynth 12.2 / 3.0 min; Random Search 42.4 / 19.1; Optuna 49.8 / 3.3 |
| Selected models | Covertype: ExtraTrees + interactions (by T2_CMAES). Adult: XGBoost, no FE (by T2_PSO) |
| Ablation (Covertype, 500 each) | EvoSynth 82.35 = PSO 82.35 = CMA-ES 82.35 > DE 82.34 > GA 82.21 (search CV) |
| Covertype run decisions | DE + CMA-ES promoted; GA swapped in (UCB 0.592 vs 0.369); rolled back after 5 iterations; 3 elite exchanges |
| Speed-up (40 configs, 12 threads) | 6.10× (parallel 5.85×, early stopping alone 1.17×) |
| Inference | Covertype ExtraTrees 12.5 ms/row, 470 MB; Adult XGBoost 0.26 ms/row, 0.4 MB |
| NAS | Fashion-MNIST, 160 evaluations, 81.8% validation |
| Constants | 40/60 split; c = 0.1; swap margin 0.10; grace 5 iterations; cooldown 2k; elite 20%; injection 50% (T1) / 20% (T2); stagnation limit 3 |
| Composite weights | Fitness 0.30, speed 0.25, efficiency 0.15, diversity 0.15, trend 0.15 |

---

## 3. Why these datasets

**Why not synthetic data?** Early versions used `make_classification` and a 30%-label-noise set. Every method landed in a narrow 79–82% band, and a professor rightly asked what +0.3 points proves. Synthetic Gaussian-mixture data has smooth boundaries that any default model captures, so there is nothing for a search to find.

**Why Covertype?** A published, widely used benchmark (Blackard & Dean 1999) with non-linear interactions between elevation, distances and soil types, and strong class imbalance:

| Class | Rows |
|---|---|
| 1 Spruce/Fir | ~211,840 |
| 2 Lodgepole Pine | ~283,301 |
| 3 Ponderosa Pine | ~35,754 |
| 4 Cottonwood/Willow | ~2,747 |
| 5 Aspen | ~9,493 |
| 6 Douglas-fir | ~17,367 |
| 7 Krummholz | ~20,510 |

Model family and feature engineering matter a lot here: untuned boosting scores about 85%, tuned ExtraTrees with interactions about 94%.

**Why Adult?** A second, very different problem: binary, mixed categorical and numeric features, scored by ROC AUC, and boosting wins. It checks that EvoSynth does not just "always pick ExtraTrees", and it is honest about easy datasets, where untuned LightGBM is within 0.06 points of every tuned method.

---

## 4. The data protocol in one picture

```
Covertype (581,012 rows), split once with a fixed seed
├── test   20,000  ── split off FIRST · never searched, never used for selection · scored once
├── search 20,000  ── (10,000 used in the smoke run) 2-fold CV for every candidate
│                     then 5-fold CV for the top 5 (confirm before select)
└── train  60,000  ── one refit of the selected configuration
```

- The search scores **hundreds** of candidates cheaply; only the **one** selected configuration is refitted on more data and tested. This is the standard multi-fidelity trade-off (FLAML and Auto-sklearn do the same), because 500 cross-validated evaluations on 581k rows would take days on a laptop.
- Random Search and Optuna go through exactly the same confirmation and refit, so all tuned methods are compared on equal terms.
- The feature engineer is fitted on training folds only and applied to validation and test data.

---

## 5. Questions and answers

### A. Novelty, motivation and "is this overkill?"

**Q1. What is novel? Isn't this just another hyperparameter tuner?**
Optuna, Hyperopt and FLAML each commit to one search strategy for the whole budget. EvoSynth is a meta-optimizer over a portfolio of four population-based optimizers:
- it scores them online (composite of fitness, speed, efficiency, diversity, trend, plus a UCB1 bonus);
- it reallocates the budget by promotion, live SWAP with rollback, stagnation injection and elite migration between algorithm families;
- model family and feature engineering are genes of the same search;
- every meta-decision is logged and shown live.

**Q2. Your faculty said this is overkill and that modern AutoML tools already do this. What is your answer?**
Partly agree, and say so. On easy datasets like Adult, defaults are already near the top and FLAML was best. The measurements show EvoSynth *ties* strong baselines rather than beating them. The contribution is not "a better tuner than Optuna"; it is an **evolutionary-algorithm portfolio controller**: a study of whether a bandit can allocate budget among GA, PSO, DE and CMA-ES online. The ablation shows it matches the best single optimizer without knowing which one that will be, and the best one changes from run to run (CMA-ES and DE in v1, PSO now). That question is not answered by existing tools, and the planned comparison with Nevergrad's NGOpt (an existing portfolio optimizer) on surrogate benchmarks addresses it directly.

**Q3. Why not just ensemble, as AutoGluon or H2O do?**
Ensembles act at prediction time: every query runs all M models, which multiplies latency and memory. EvoSynth uses several *search algorithms* offline and ships one model, so serving costs exactly what that model costs. Be precise about that cost: it is model-dependent (0.26 ms for the Adult XGBoost, 12.5 ms and 470 MB for the Covertype ExtraTrees). Whether a stacked ensemble would be more accurate on these splits has not been measured.

**Q4. How does EvoSynth deal with the No Free Lunch theorem?**
It does not claim to beat it. It hedges: GA explores broadly, PSO exploits quickly, DE explores robustly, CMA-ES fine-tunes, and the bandit shifts budget toward whichever is making progress on *this* problem.

**Q5. How is it different from algorithm portfolios or hyper-heuristics?**
It combines a population-level portfolio, a two-tier structure separating exploration (Tier 1) from exploitation (Tier 2), a composite reward including diversity and trend, and the transfer of actual solutions between algorithm families. It also searches model selection and feature engineering jointly.

### B. Data and validation

**Q6. You searched on only 10,000 rows. How can you claim results on Covertype?**
The search only *ranks* candidates. The selected configuration is refitted on 60,000 rows and scored once on 20,000 test rows that nothing else ever saw. Every baseline is scored on the same test rows.

**Q7. Is there data leakage?**
No. The test rows are split off first with a fixed seed; search and train rows come from the rest; the feature engineer is fitted on training folds only; confirmation uses search rows only. The exact row indices are published so anyone can check.

**Q8. Why is test accuracy (93.89%) so much higher than search CV (82.35%)?**
Training-set size. Each search fold trains on 5,000 rows with forests capped at 150 trees; the confirmation trains on 8,000 rows per fold (84.84%); the refit trains on 60,000. Forests on Covertype improve strongly with more data. Random Search and Optuna show the same gap.

**Q9. Why is a high accuracy on Covertype harder than it looks?**
Two classes make up more than 84% of rows, so a model can do well on accuracy while failing the minority classes (Aspen, Cottonwood/Willow, Douglas-fir). Going from 85% (untuned XGBoost) to 94% means classifying many of those correctly.

**Q10. What is "confirm before select", and why add it?**
Choosing the maximum of hundreds of noisy 2-fold scores favours lucky configurations (overtuning, the AutoML 2025 best paper's topic). The top 5 distinct configurations are re-scored with 5-fold CV and the best confirmed one is chosen. On Adult it changed the choice (search rank 2 was selected).

### C. Mechanics and maths

**Q11. Explain the composite score. Why not reward accuracy alone?**
An island can hold a high best while its population has collapsed and stopped improving; a pure-accuracy bandit would keep feeding it. Each term is min–max normalised across families:

`0.30·fitness + 0.25·speed + 0.15·efficiency + 0.15·diversity + 0.15·trend`

- **Speed:** improvement of the family's best since its last scoring call.
- **Efficiency:** that improvement per second of evaluation time.
- **Diversity:** mean pairwise distance between the island's latent vectors, divided by √d.
- **Trend:** slope of the best over the last 3 scoring calls.

In the Covertype run DE was promoted over PSO and GA, and scored far above CMA-ES (composite 0.961 vs 0.417), even though CMA-ES had the higher best score, because DE was still improving and had the most diverse island.

**Q12. What is the UCB term doing?**
`U = composite + 0.1·sqrt(ln N_total / n_i)`, the UCB1 exploration bonus, applied by the `UCB1Bandit`. It grows for families that have used fewer evaluations, so one that was starved early still gets a chance. With c = 0.1 it mostly breaks near-ties (about 0.02–0.05 in the runs).

**Q13. What triggers a SWAP, and what happened in your run?**
At each checkpoint, a SWAP happens when the best Tier 1 family's U exceeds the weaker Tier 2 family's U by more than 0.10, the challenger is not cooling down, and no grace period is active. The challenger takes over the Tier 2 island; the demoted family gets a 2k cooldown; a 5-iteration grace period starts, and if the global best does not improve, the swap is rolled back from a snapshot.

In the Covertype run: GA (U 0.592) replaced DE (U 0.369). GA reached the 82.35% plateau through elites shared by CMA-ES, but a tie is not an improvement, so after 5 iterations the swap was **rolled back** and DE returned. In the Adult run two swaps happened (CMA-ES replaced GA, then DE replaced CMA-ES).

**Q14. What is the universal latent space, and why is it needed?**
PSO velocities, DE differences, CMA-ES covariance and GA blending all need continuous vectors. Every hyperparameter maps to `[0,1]`: continuous values linearly or logarithmically, integers rounded, categoricals with K options into `min(⌊zK⌋, K−1)` bins. Every optimizer can then move through "XGBoost vs ExtraTrees" as easily as through a learning rate.

**Q15. How does the feature engineering stay dataset-agnostic?**
`GenericFeatureEngineer` uses no column names: columns with more than 15 distinct values count as continuous; `stats` adds row mean, standard deviation, minimum and maximum; `interactions` also adds pairwise differences and Euclidean norms of the highest-variance continuous columns. The strategy is a gene: the search picked `interactions` on Covertype and `none` on Adult.

**Q16. How do you prevent ping-ponging and bad promotions?**
Cooldown (2k iterations before a demoted family can return), the grace period with rollback, the 0.10 margin, and only one pending swap at a time.

**Q17. What happens to stagnating islands?**
A Tier 1 island whose best did not improve has 50% of its members replaced by random vectors at once; a Tier 2 island stagnant for 3 consecutive checkpoints has its worst 20% replaced. Injection goes through each optimizer's `inject` method, so PSO velocities, CMA-ES state and DE targets are updated too. Elite sharing runs only between the two Tier 2 islands, so Tier 1 stays a diverse sandbox.

**Q18. Can a demoted optimizer return? Does the population size change?**
Yes, all four stay alive and a demoted one can be swapped back after its cooldown. The total number of evaluations is fixed by the budget; island sizes are fixed per tier; injection replaces members rather than adding them.

### D. Performance and trade-offs

**Q19. Doesn't running four algorithms cost four times as much?**
No. The budget is a fixed number of evaluations shared by all optimizers. Scoring and swapping cost microseconds; almost all the time is model training, as in any tuner. In the ablation, EvoSynth's 500 evaluations took about as long as GA's 500 alone.

**Q20. How did you speed it up, and by how much?**
Batches are evaluated in a thread pool (model libraries release the GIL), boosted models stop early, and forests are capped at 150 trees during the search. Measured on 40 Covertype configurations with 12 threads: **6.1×**, almost all from parallelism (early stopping alone gives 1.17×). The old "11.5×" was never measured by code and should not be quoted. Note that Optuna can also run trials in parallel, so this is an engineering gain, not a property of the search method.

**Q21. Default models score 85–90% in seconds. Why spend 12 minutes?**
On Covertype the gain is 4.1 points over the best untuned model (about 820 more test rows correct), and the cost is paid once, offline. On Adult the gain is only 0.03–0.06 points, and saying so is part of a credible answer: tuning pays off on some datasets and barely on others.

### E. The hard questions (answer honestly)

**Q22. Random Search and Optuna scored the same as EvoSynth. Isn't EvoSynth unnecessary?**
On Covertype, yes, the methods tie, and for a structural reason: the five best configurations decode to *the same effective model* (forests capped at 150 trees, the same depth and the same number of features), so every competent search hits the same 82.35% plateau and nothing can beat it. That makes Covertype a poor test for separating optimizers. EvoSynth's measurable advantages are elsewhere: it was 3.5–4× faster on Covertype through parallel evaluation, and it matches the best single optimizer without knowing which one that is. Harder datasets (HIGGS, Poker Hand), more seeds and surrogate benchmarks are the planned next step.

**Q23. FLAML won Adult. Why?**
FLAML searched on all 36,000 train rows with its own holdout (EvoSynth searched on 10,000) and found a 1,861-tree XGBoost with a very low learning rate. On an easy dataset where boosting with defaults is already near the top, a fast, cost-frugal tool with more training data wins. EvoSynth's design targets harder, multi-modal landscapes, which Adult is not.

**Q24. In the ablation, PSO alone was ahead at 125 evaluations. Why use EvoSynth?**
Because you only know *which* single optimizer wins after the run, and it changes: PSO here, CMA-ES and DE in v1, and GA was the worst here. EvoSynth reached the same final score (82.35%) without that hindsight. With one seed and the Covertype plateau, the honest claim is "matches the best single optimizer", not "beats it".

**Q25. What changed from the numbers you presented at the panel review?**
The v1 results had bugs and an unfair protocol: the test rows could overlap the search sample; Optuna and Random Search were refitted with default hyperparameters (which made EvoSynth look 4 points better than Optuna); the ablation's EvoSynth diversity and UCB values were constants; the untuned floors had no generating script; 11.5× and "2 ms / 150 MB" were not measured. Phases 0–4 fixed all of these and Phase 5 re-ran everything. Treat the change as a strength: the current numbers are reproducible from published splits.

**Q26. What are the limitations?**
- Ties rather than beats strong baselines on the current datasets; FLAML won Adult.
- One seed so far.
- The Covertype search space has a plateau.
- The search optimises accuracy only, so the Covertype winner is a large, slow forest.
- Tabular data and small CNNs only; single machine; no successive halving.
- Boosting-only genes are inactive for forests.

**Future work:** the multi-seed `lean` plan; HIGGS and Poker Hand; an NGOpt baseline and surrogate benchmarks; rank-normalised rewards; a latency- or size-aware objective; Ray workers.

**Q27. Why SSE rather than WebSockets? What does the dashboard show?**
Telemetry flows one way; SSE is simpler, reconnects automatically and passes through firewalls. The dashboard shows the workflow step and budget, which optimizers are in each tier, each family's best score, every promotion, swap, rollback, injection and elite exchange, and the full log with the UCB scoring tables.

---

## 6. Rival comparison (the battlecard)

| | Optuna | FLAML | Auto-sklearn 2.0 | **EvoSynth** |
|---|---|---|---|---|
| Search | One sampler (TPE / CMA-ES) per study | Cost-frugal local search | Bayesian optimization + meta-learning | **UCB1 bandit over 4 optimizers** |
| Strategy switching | No | No | No | **Live SWAP with rollback** |
| Knowledge sharing between algorithms | No | No | No | **Pairwise elite migration** |
| Feature engineering | Manual | Limited | Fixed preprocessors | **Searchable gene** |
| Output | One model | One model | Ensemble | **One model** |
| Measured test (Covertype / Adult) | 93.87 / 92.95 | 92.53 / 93.13 | not run | **93.89 / 92.96** |
| Observability | Dashboard (polling) | Logs | Tables | **Live SSE dashboard** |

The longer matrix, with AutoGluon and H2O, is in [09](09-comparative-analysis.md) §4.

---

## 7. Strengths and limitations to state yourself

**Emphasise:**
1. A portfolio of four evolutionary and swarm optimizers with online budget allocation, which matches the best single optimizer without choosing it in advance.
2. Model family and feature engineering searched together; the right family on both datasets.
3. A fair, reproducible protocol: held-out test rows, confirm before select, identical refit, published splits.
4. Measured speed-up (6.1×) and measured inference cost.
5. Every meta-decision visible live and recorded in the run's decision log.

**Acknowledge before you are asked:** the ties with Random Search and Optuna, FLAML winning Adult, one seed, the Covertype plateau, and the size of the Covertype model.

---

## 8. Checklist before a review

- [ ] `py -3.14 -m pytest -q` passes (88 tests).
- [ ] `results/summary/` is current (`evosynth summary`); the numbers on your slides match `main_table.md`.
- [ ] `paper/figures/approval.png` for this review is in place.
- [ ] `.data/covertype/` and `.data/adult/` are cached (`evosynth datasets download covertype adult`), so a live demo does not wait on a download.
- [ ] Demo command tested: `evosynth run --full --dataset adult --subsample 10000` (about 3 minutes) with the dashboard open.
- [ ] You can explain the Covertype plateau, the rollback in the Covertype run, and why FLAML won Adult.

---

## Quiz

**1.** A panel member says "EvoSynth is no better than Optuna." What is the accurate reply?

<details><summary>Answer</summary>

On accuracy, that is right for these datasets: under the same confirm-and-refit protocol they tie (93.89 vs 93.87 on Covertype, 92.96 vs 92.95 on Adult). EvoSynth's contributions are the evolutionary portfolio (it matches the best single optimizer without knowing which one in advance), joint model and feature selection, and a shorter wall time from parallel evaluation.
</details>

**2.** Why can no method beat 82.35% search CV on Covertype?

<details><summary>Answer</summary>

The best configurations all decode to the same effective ExtraTrees model during the search: trees are capped at 150, depth 9 maps to 27, the column-sample ratio rounds to the same number of features, and the boosting-only genes are ignored. With the same random seed they are identical models, so 82.35% is the ceiling of that region.
</details>

**3.** What happened to GA in the Covertype run, and why?

<details><summary>Answer</summary>

GA was swapped into Tier 2 (UCB 0.592 beat DE's 0.369 by more than 0.10) because it was improving quickly. It only tied the global best, through CMA-ES's shared elites, during its 5-iteration grace period, so the swap was rolled back and DE returned.
</details>

**4.** What is the measured speed-up, and where does it come from?

<details><summary>Answer</summary>

6.1× on 12 threads for 40 Covertype configurations. Almost all of it comes from parallel evaluation (5.85×); early stopping alone gives 1.17×.
</details>

**5.** Give two flaws of the panel-review (v1) results that Phases 0–5 fixed.

<details><summary>Answer</summary>

Any two of: test rows could overlap the search sample; Optuna and Random Search were refitted with default hyperparameters; the ablation's EvoSynth diversity and UCB values were constants; the untuned floors had no generating script; the speed-up and inference-cost claims were never measured.
</details>

**6.** Why was Adult added alongside Covertype?

<details><summary>Answer</summary>

It is a different kind of problem (binary, mixed features, ROC AUC) where boosting wins rather than forests. It shows EvoSynth chooses the model family per dataset, and it shows honestly that on easy datasets tuning adds little (untuned LightGBM is within 0.06 points).
</details>

**7.** What does "confirm before select" protect against, and did it change anything?

<details><summary>Answer</summary>

Overtuning: choosing a configuration because its noisy 2-fold score is high by chance. The top 5 are re-scored with 5-fold CV. On Adult it selected the configuration ranked second in the search.
</details>

**8.** Where does EvoSynth's shorter wall time on Covertype come from, and is it a fair advantage to claim?

<details><summary>Answer</summary>

From evaluating up to 8 candidates in parallel, while the baselines ran one trial at a time. It is a real advantage of the implementation, but not of the search method: Optuna can also run trials in parallel. Present it that way.
</details>
