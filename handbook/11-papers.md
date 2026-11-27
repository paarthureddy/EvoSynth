# 11 — Papers Behind EvoSynth

Every method, baseline, model, dataset and library that the paper (`paper/paper.tex`) and the review report (`paper/report.tex`) use is backed by a citation. This chapter lists all 47 references once. For each one it gives what the work says and how EvoSynth uses it. The review slides (`paper/slides.tex`) list only the ten recent papers of section 1.

**How the reference lists are organised.** Each document's bibliography is split into groups. Within a group, references appear in the order they are first cited.

| Group | What goes in it |
|---|---|
| Research Papers | Journal articles |
| Conference Papers | Papers in conference or workshop proceedings |
| Books and Book Chapters | Books and chapters in edited books |
| Preprints | arXiv papers that have not appeared in a journal or conference |

The **Cited in** column below says which documents list the reference (P = paper, R = report, S = slides).

| Document | References | What it lists |
|---|---|---|
| Paper | 47 | Everything it uses or discusses, including related work only the paper covers |
| Report | 37 | Everything it uses, plus the ten recent papers of its literature review |
| Slides | 10 | Only the ten recent papers (the literature review) |

**What is deliberately not cited.**
- Textbook concepts: cross-validation, ROC AUC, stacking and convolutional neural networks.
- Datasets EvoSynth supports but did not use in the reported experiments: HIGGS, Poker Hand, MiniBooNE, Airlines and Jannis.
- Tools mentioned only as future work, such as Ray.
- FastAPI, React, Vite, Tailwind and ConfigSpace, which have no citable paper.

---

## 1. The ten recent papers (2024–2026): the literature review

These ten form the literature review in the report (one table row each), the core of the related-work section in the paper, and the whole reference slide of the deck. Each one was checked against the paper's own text (abstract, introduction and conclusion) in November 2026.

| Reference | What it says | How EvoSynth uses it | Cited in |
|---|---|---|---|
| **Rezaei Balef, Vernade, Eggensperger (2025)**, *Put CASH on Bandits: A Max K-Armed Problem for AutoML*, NeurIPS 2025. Conference. | Treats CASH as a bandit problem. Each arm is one model class with its own HPO run, and the MaxUCB algorithm picks which arm gets the next HPO iteration. Because the arms care about the *best* score rather than the average, it is a "max k-armed" bandit. It beats searching the combined space on four AutoML benchmarks. Its stated limitation is that it assumes rewards are stationary. | The closest recent work. Both use a UCB-style bandit to split a tuning budget. The difference is what an arm is. In MaxUCB an arm is a **model class** with one fixed tuner. In EvoSynth an arm is an **optimizer** (GA, PSO, DE, CMA-ES), and every optimizer searches the same joint space of models, feature engineering and hyperparameters. Named in future work as a baseline to compare against. | P, R, S |
| **Guo et al. (2024)**, *Deep Reinforcement Learning for Dynamic Algorithm Selection: A Proof-of-Principle Study on Differential Evolution*, IEEE Trans. SMC: Systems 54(7). Journal. | RL-DAS: a deep-RL agent, trained offline with PPO, switches between several DE variants during one run, based on landscape and algorithm features. A context memory restores each variant's internal state when it is switched back in (a "warm start"). Dynamic switching improved overall performance and generalised across problem classes. | Strongest evidence that switching optimizers *during* a run pays off. RL-DAS's context restoration is the same idea as EvoSynth's SWAP with rollback, which saves a snapshot and can restore it. Two contrasts: EvoSynth needs **no offline training** (UCB1 learns within the run), and it switches between different families rather than variants of one. | P, R, S |
| **Ma, Guo, Gong, Zhang, Tan (2025)**, *Toward Automated Algorithm Design: A Survey and Practical Guide to Meta-Black-Box-Optimization*, IEEE Trans. Evol. Comput. Journal. | Surveys "MetaBBO": using learning (RL, supervised, neuroevolution, LLMs) to select, configure or generate optimization algorithms. Gives a taxonomy (selection, configuration, solution manipulation, generation) and benchmarks representative methods. | Places EvoSynth within *dynamic algorithm selection*. EvoSynth sits at the simple end: an online bandit rather than a learned meta-policy. | P, R, S |
| **Črepinšek et al. (2025)**, *Overcoming Stagnation in Metaheuristic Algorithms with MsMA's Adaptive Meta-Level Partitioning*, Mathematics 13(11). Journal. | MsMA wraps any metaheuristic, watches it from outside, and reallocates its remaining evaluation budget when progress stalls. It improved every algorithm tested on CEC'24 and on a real load-flow problem. | Supports EvoSynth's stagnation handling: random members are injected into islands that stop improving, and budget is moved away from stalled optimizers. MsMA does this for one algorithm on benchmark functions; EvoSynth does it across four optimizers on ML pipelines. | P, R, S |
| **Franceschi et al. (2025)**, *Hyperparameter Optimization in Machine Learning*, Foundations and Trends in ML 18(6). Journal. | A 140-page monograph covering random and quasi-random search, and bandit-, model-, population- and gradient-based HPO, plus online, constrained and multi-objective HPO. | General reference for what HPO is and for the families EvoSynth combines (population-based and bandit-based). | P, R, S |
| **Alcobaça & de Carvalho (2026)**, *A Literature Review on Automated Machine Learning*, Artificial Intelligence Review 59. Journal. | A systematic review of AutoML literature reviews, covering meta-learning, HPO and transfer learning, with applications and research directions. | Background survey for the AutoML field in the report and paper. | P, R, S |
| **Schneider, Bischl, Feurer (2025)**, *Overtuning in Hyperparameter Optimization*, AutoML Conf 2025. Conference. | Defines **overtuning**: optimizing a noisy validation score too hard picks a configuration that generalizes worse. A reanalysis of large HPO benchmark data finds it common, usually mild, and in about 10% of cases the chosen configuration is worse than the default. | The reason for **confirm-before-select**: EvoSynth re-scores the five best distinct configurations with 5-fold CV before choosing, instead of trusting the single best noisy 2-fold score. | P, R, S |
| **Nagler, Schneider, Bischl, Feurer (2024)**, *Reshuffling Resampling Splits Can Improve Generalization of HPO*, NeurIPS 2024. Conference. | Shows theoretically and empirically that drawing new train/validation splits for each configuration often generalizes better than fixed splits, and can make cheap holdout as good as CV. | Second backing for the confirmation step. How configurations are scored and selected changes what you get on test data, so EvoSynth re-scores finalists with more folds and keeps test rows untouched. | P, R, S |
| **Erickson et al. (2025)**, *TabArena: A Living Benchmark for ML on Tabular Data*, NeurIPS 2025 Datasets & Benchmarks. Conference. | A continuously maintained benchmark of 16 models on 51 tabular datasets. Tuning plus post-hoc ensembling (per model and across models) gives the best accuracy; gradient-boosted trees remain strong. | Evidence for the single-model trade-off: top tabular accuracy comes from ensembles, which cost inference time and memory. EvoSynth delivers one model and measures its cost instead. | P, R, S |
| **Benjamins et al. (2025)**, *carps: A Framework for Comparing N Hyperparameter Optimizers on M Benchmarks*, arXiv. Preprint. | A framework with 3,336 HPO tasks and 26 optimizer variants, plus small representative task subsets for fast, fair comparisons. | Named in future work: the many-task benchmark on which EvoSynth should be compared with NGOpt and MaxUCB, answering the "two datasets, one seed" limitation. | P, R, S |

---

## 2. Hyperparameter optimization and AutoML foundations

| Reference | What it says | How EvoSynth uses it | Cited in |
|---|---|---|---|
| **Feurer & Hutter (2019)**, *Hyperparameter Optimization*, chapter in *Automated Machine Learning* (Springer). Book chapter. | Standard definition of HPO as black-box optimization of a validation score, with an overview of methods. | Definition of the problem in the paper's introduction and the report's problem statement. | P, R |
| **Bischl et al. (2023)**, *Hyperparameter Optimization: Foundations, Algorithms, Best Practices, and Open Challenges*, WIREs DMKD 13(2). Journal. | A review of HPO families and practical advice on resampling, budgets and evaluation. | Background for HPO and for EvoSynth's evaluation practice (held-out test rows, resampling during search). | P, R |
| **Thornton et al. (2013)**, *Auto-WEKA*, KDD 2013. Conference. | Defines **CASH** (combined algorithm selection and hyperparameter optimization) and solves it as one hierarchical search. | EvoSynth solves CASH. The model family and feature-engineering strategy are genes in the same latent vector as the hyperparameters. | P, R |
| **Bergstra & Bengio (2012)**, *Random Search for Hyper-Parameter Optimization*, JMLR 13. Journal. | Random search beats grid search when only a few hyperparameters matter, and is a strong baseline. | The **Random Search** baseline: 500 trials, same space, same evaluator, same confirm-and-refit. | P, R |
| **Bergstra et al. (2011)**, *Algorithms for Hyper-Parameter Optimization*, NeurIPS 2011. Conference. | Introduces **TPE** (tree-structured Parzen estimator) Bayesian optimization. | TPE is the sampler of the **Optuna TPE** baseline. | P, R |
| **Hutter, Hoos, Leyton-Brown (2011)**, *SMAC*, LION 5. Conference. | Bayesian optimization with random-forest surrogates for algorithm configuration. | Example of the "single static search strategy" (Bayesian optimization) that EvoSynth contrasts with. | P, R |
| **Li et al. (2018)**, *Hyperband*, JMLR 18. Journal. | Bandit-based multi-fidelity HPO that spends budget on configurations by successive halving. | Related work: a bandit over *configurations*, whereas EvoSynth's bandit is over *optimizers*. | P |
| **Feurer et al. (2015)**, *auto-sklearn*, NeurIPS 2015. Conference. | AutoML system with meta-learned warm starts and post-hoc ensembles. | Related work: a single-strategy AutoML system that ends in an ensemble. | P |

---

## 3. The four optimizers in the portfolio

| Reference | What it says | How EvoSynth uses it | Cited in |
|---|---|---|---|
| **Holland (1975)**, *Adaptation in Natural and Artificial Systems*. Book. | Founding work on **genetic algorithms**: populations, selection, crossover, mutation. | The **GA** optimizer (tournament selection, crossover and mutation on latent vectors). | P, R |
| **Kennedy & Eberhart (1995)**, *Particle Swarm Optimization*, ICNN'95. Conference. | **PSO**: particles move under inertia and attraction to personal and global bests. | The **PSO** optimizer. Continuous, so it relies on EvoSynth's latent space to handle categorical genes. | P, R |
| **Storn & Price (1997)**, *Differential Evolution*, J. Global Optimization 11. Journal. | **DE**: mutation from scaled differences of population members, then one-to-one selection. | The **DE** optimizer (one-to-one replacement, round-robin targets). | P, R |
| **Hansen & Ostermeier (2001)**, *Completely Derandomized Self-Adaptation in Evolution Strategies*, Evolutionary Computation 9(2). Journal. | **CMA-ES**: samples from a multivariate normal and adapts its covariance matrix to the landscape. | The **CMA-ES** optimizer, through the `cma` library. It found the winning Covertype configuration. | P, R |

---

## 4. Bandits, algorithm selection and portfolios

| Reference | What it says | How EvoSynth uses it | Cited in |
|---|---|---|---|
| **Wolpert & Macready (1997)**, *No Free Lunch Theorems for Optimization*, IEEE TEVC 1(1). Journal. | Averaged over all problems, every optimizer performs the same, so none is best everywhere. | The project's motivation: since the best optimizer for a dataset cannot be known in advance, EvoSynth runs several and moves budget online. | P, R |
| **Auer, Cesa-Bianchi, Fischer (2002)**, *Finite-Time Analysis of the Multiarmed Bandit Problem*, Machine Learning 47. Journal. | Introduces **UCB1** with its exploration bonus `sqrt(2 ln n / n_i)` and its regret bound. | EvoSynth's bandit: `UCB_i = CS_i + c·sqrt(ln E_total / E_i)` with `c = 0.1`. The composite score replaces the mean reward. | P, R |
| **Rice (1976)**, *The Algorithm Selection Problem*, Advances in Computers 15. Book chapter (serial volume). | Formal statement of choosing the best algorithm for a problem instance. | Frames EvoSynth's question: which optimizer for this dataset, answered online. | P |
| **Kerschke et al. (2019)**, *Automated Algorithm Selection: Survey and Perspectives*, Evolutionary Computation 27(1). Journal. | Survey of algorithm selection learned offline from instance features. | The offline approach EvoSynth contrasts with: EvoSynth needs no instance features or training set. | P |
| **Fialho et al. (2010)**, *Analyzing Bandit-Based Adaptive Operator Selection Mechanisms*, Annals Math. AI 60. Journal. | Uses UCB-style bandits to pick *variation operators* inside one EA, with credit assignment. | The closest older idea. EvoSynth moves the bandit one level up, from operators to whole optimizers. | P |
| **Peng et al. (2010)**, *Population-Based Algorithm Portfolios*, IEEE TEVC 14(5). Journal. | Splits a fixed budget among several EAs whose populations exchange individuals. | Direct ancestor of EvoSynth's portfolio-with-migration design. | P |
| **Vrugt & Robinson (2007)**, *AMALGAM*, PNAS 104(3). Journal. | Several search methods produce offspring, and each method's share adapts to its success. | Another adaptive multi-method ancestor; EvoSynth adapts shares with UCB1 and tiers instead. | P |
| **Meunier et al. (2022)**, *Black-Box Optimization Revisited (NGOpt)*, IEEE TEVC 26(3). Journal. | Nevergrad's NGOpt picks an optimizer with rules derived from massive benchmarking. | Rule-based (offline) optimizer selection. Named as a future comparison baseline. | P, R |

---

## 5. Island models and population-based search

| Reference | What it says | How EvoSynth uses it | Cited in |
|---|---|---|---|
| **Alba & Tomassini (2002)**, *Parallelism and Evolutionary Algorithms*, IEEE TEVC 6(5). Journal. | Island models: separate populations with occasional migration keep diversity and parallelise well. | EvoSynth's two-tier island model. Each island belongs to a different optimizer, and the Tier 2 islands exchange their top 20% pairwise. | P, R |
| **Jaderberg et al. (2017)**, *Population Based Training of Neural Networks*, arXiv. Preprint. | Evolves hyperparameters of a population of models during training (exploit and explore). | Related population-based HPO. | P |
| **Real et al. (2019)**, *Regularized Evolution for Image Classifier Architecture Search*, AAAI 2019. Conference. | Aging evolution is a strong, simple method for NAS. | Background for running the same evolutionary engine on a CNN search space (the Fashion-MNIST NAS experiment). | P |
| **Zoph & Le (2017)**, *Neural Architecture Search with Reinforcement Learning*, ICLR 2017. Conference. | Introduces **NAS**: automatically searching network architectures. | Backs the term NAS for EvoSynth's CNN experiment. | P, R |

---

## 6. AutoML systems and the ensembling contrast

| Reference | What it says | How EvoSynth uses it | Cited in |
|---|---|---|---|
| **Akiba et al. (2019)**, *Optuna*, KDD 2019. Conference. | Define-by-run HPO framework with TPE and CMA-ES samplers and pruning. | **Optuna TPE** baseline: 500 trials, same space and evaluator, same confirm-and-refit. | P, R |
| **Wang et al. (2021)**, *FLAML: A Fast and Lightweight AutoML Library*, MLSys 2021. Conference. | Cost-frugal local search with learner selection, giving accurate models at low compute. | **FLAML** baseline, given EvoSynth's wall time. It was best on Adult (+0.17). | P, R |
| **LeDell & Poirier (2020)**, *H2O AutoML*, ICML AutoML Workshop. Conference (workshop). | Random search plus stacked ensembles of many models. | The ensembling system EvoSynth contrasts with (high accuracy, high inference cost). | P, R |
| **Erickson et al. (2020)**, *AutoGluon-Tabular*, arXiv. Preprint. | Multi-layer stacking and bagging of many models for tabular data. | Second example of the stacking approach. | P |

---

## 7. Models in the search space

| Reference | What it says | How EvoSynth uses it | Cited in |
|---|---|---|---|
| **Chen & Guestrin (2016)**, *XGBoost*, KDD 2016. Conference. | Scalable regularised gradient-boosted trees. | Model-family gene; selected on Adult. | P, R |
| **Ke et al. (2017)**, *LightGBM*, NeurIPS 2017. Conference. | Fast gradient boosting with histogram splits and leaf-wise growth. | Model-family gene. | P, R |
| **Breiman (2001)**, *Random Forests*, Machine Learning 45. Journal. | Bagged decision trees with random feature subsets. | Model-family gene. | P, R |
| **Geurts, Ernst, Wehenkel (2006)**, *Extremely Randomized Trees*, Machine Learning 63. Journal. | Trees with random split thresholds; lower variance, fast to train. | Model-family gene; selected on Covertype (with interaction features). | P, R |

---

## 8. Datasets

| Reference | What it says | How EvoSynth uses it | Cited in |
|---|---|---|---|
| **Blackard & Dean (1999)**, Computers and Electronics in Agriculture 24(3). Journal. | Origin of the **Forest Covertype** data (cartographic variables, 7 cover types). | Main dataset: 581,012 rows, 54 features, accuracy. | P, R |
| **Kohavi (1996)**, *Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid*, KDD 1996. Conference. | Origin of the **Adult** (census income) data. | Second dataset: 48,842 rows, binary, ROC AUC. | P, R |
| **Xiao, Rasul, Vollgraf (2017)**, *Fashion-MNIST*, arXiv. Preprint. | 70,000 28×28 clothing images in 10 classes, a harder drop-in for MNIST. | NAS experiment (81.8% validation accuracy with 160 evaluations). | P, R |

---

## 9. Software

| Reference | What it says | How EvoSynth uses it | Cited in |
|---|---|---|---|
| **Pedregosa et al. (2011)**, *Scikit-learn: Machine Learning in Python*, JMLR 12. Journal. | The scikit-learn library. | Random forest, extra trees, cross-validation and preprocessing. | P, R |
| **Paszke et al. (2019)**, *PyTorch*, NeurIPS 2019. Conference. | The PyTorch deep-learning library. | CNN evaluator for NAS. | P, R |

---

## Quiz

<details><summary>1. Which recent paper is closest to EvoSynth, and what is the key difference?</summary>

MaxUCB (Rezaei Balef et al., NeurIPS 2025). Both use a UCB-style bandit to split an HPO budget. In MaxUCB the arms are model classes, each with one fixed tuner; in EvoSynth the arms are four optimizers that all search the same joint model + hyperparameter space.
</details>

<details><summary>2. Which two papers justify confirm-before-select?</summary>

Schneider et al. 2025 (overtuning) and Nagler et al. 2024 (reshuffling resampling splits). Both show that picking the best of many noisy validation scores can generalize worse, so EvoSynth re-scores the top 5 with 5-fold CV.
</details>

<details><summary>3. How does EvoSynth differ from RL-DAS?</summary>

RL-DAS trains a deep-RL agent offline to switch among DE variants. EvoSynth's UCB1 bandit learns within the run (no offline training) and switches among four different optimizer families.
</details>

<details><summary>4. Why do the paper's references include TabArena and AutoGluon if EvoSynth does not ensemble?</summary>

They back the claim that top tabular accuracy comes from ensembles. EvoSynth deliberately delivers one model and measures its inference cost, so these papers define the trade-off it takes.
</details>
