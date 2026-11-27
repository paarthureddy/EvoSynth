# Introduction to EvoSynth

## What is EvoSynth?

**EvoSynth** is a **Dynamic Multi-Armed Bandit Meta-Optimizer for Automated Model Selection and Hyperparameter Tuning**.

Traditional Automated Machine Learning (AutoML) systems suffer from one of two severe limitations:
1. **Single Static Algorithm Trap:** They run one fixed optimizer (e.g., standard Bayesian Optimization or localized hill climbing). According to the **No Free Lunch Theorem**, no single optimizer excels across all loss surfaces. On rugged, multi-modal, or noisy landscapes, static algorithms get permanently trapped in local minima.
2. **Ensembles at prediction time:** Systems like H2O.ai and AutoGluon often deliver **stacked ensembles**, which keep many models in memory and run all of them for every prediction.

EvoSynth resolves both dilemmas. It operates as a **meta-optimizer** during search time, maintaining a portfolio of 4 complementary evolutionary and swarm algorithms:
* **Genetic Algorithm (GA)** — Global stochastic exploration via crossover and mutation.
* **Particle Swarm Optimization (PSO)** — Particles move with momentum towards their own and the swarm's best positions.
* **Differential Evolution (DE)** — Vector difference perturbations for non-linear landscapes.
* **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)** — State-of-the-art continuous covariance estimation.

The core meta-controller uses an **Upper Confidence Bound (UCB1)** Multi-Armed Bandit to dynamically allocate evaluation budget to the algorithm making the fastest progress, while periodically migrating elite candidate solutions across islands.

---

## Why the Project Topic Name?

The topic name reflects the 4 foundational stages of the pipeline:

```
[ EvoSynth: Dynamic Multi-Armed Bandit Meta-Optimizer for Automated Model Selection and Hyperparameter Tuning ]
   │         │                            │               │                         │
   │         │                            │               │                         └── Continuous & discrete search with Early Stopping
   │         │                            │               └── Dynamic architecture switching (XGBoost / LightGBM / RF / CNN)
   │         │                            └── Controls a portfolio of 4 competing search algorithms
   │         └── Real-time UCB1 budget allocation balancing exploration vs. exploitation
   └── System name (Evolutionary Synthesis)
```

1. **Dynamic Multi-Armed Bandit:** The UCB1 bandit ranks the optimizer families on a composite of fitness, speed, efficiency, diversity and trend, and decides which families are promoted and when they are swapped.
2. **Meta-Optimizer:** It does not merely optimize an objective function; it optimizes the execution and resource allocation of *other optimizers*.
3. **Automated Model Selection:** The search space dynamically resolves categorical architecture choices (`XGBoost`, `LightGBM`, `RandomForest`, or PyTorch `CNN`).
4. **Hyperparameter Tuning:** Fine-tunes all learning rates, tree depths, regularization, channels, and activations with parallel early stopping.

---

## Core Capabilities at a Glance

* **Multi-Model Selection:** Evaluates XGBoost, LightGBM, Random Forest and ExtraTrees in the same run, together with a searchable feature-engineering strategy.
* **Real-world datasets:** Covertype, HIGGS, Adult, Airlines, Poker Hand, MiniBooNE and Jannis, each with fixed held-out test rows and its own metric (accuracy or ROC AUC); Fashion-MNIST for NAS.
* **Robust selection:** The top configurations are re-scored with 5-fold CV before the winner is chosen, then retrained and scored once on held-out data.
* **Neural Architecture Search (NAS):** Tunes PyTorch Convolutional Neural Networks on image datasets.
* **Cooperative Co-evolution (Island Model):** Separates populations into Tier 1 (Exploration) and Tier 2 (Exploitation) islands.
* **Asynchronous Parallelism:** Evaluates candidate configurations across multi-core CPUs in parallel.
* **Early Stopping:** Terminates unpromising tree boosting runs early, using a slice of the training data.
* **Reproducible:** One seed reproduces a run exactly; every setting lives in `settings/*.toml`.
* **Glass-Box Telemetry:** Live streaming via Server-Sent Events (SSE) into a React visualization dashboard.

For measured results against Random Search, Optuna and FLAML on Covertype and Adult, see [Results](./results.md).
