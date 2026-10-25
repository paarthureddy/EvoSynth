# System Architecture

EvoSynth is structured into three clean, decoupled architectural layers:

```text
EvoSynth System Architecture
┌────────────────────────────────────────────────────────────────────────┐
│                        Meta-Control Layer                              │
│  • DynamicOptimizer (Controller)                                       │
│  • UCB1 Multi-Armed Bandit Controller                                  │
│  • Composite Scoring Engine (Fitness, Velocity, Diversity, Trend)      │
│  • Tier Manager (Tier 1 Exploration vs Tier 2 Exploitation)            │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Ask / Tell Iteration Loops
┌──────────────────────────────────▼─────────────────────────────────────┐
│                      Worker Optimizer Layer                            │
│  • GeneticAlgorithm (GA)            • DifferentialEvolution (DE)       │
│  • ParticleSwarm (PSO)              • CMAESOptimizer (CMA-ES)          │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Parallel Vector Evaluation
┌──────────────────────────────────▼─────────────────────────────────────┐
│                     Evaluation & Search Space Layer                    │
│  • Universal Latent Space Continuous Mapper ([0, 1]^n)                 │
│  • MultiModelEvaluator (XGBoost, LightGBM, Random Forest)              │
│  • ImageCNNEvaluator (PyTorch CNN Neural Architecture Search)          │
│  • PathologicalXGBoostEvaluator (Noisy Local Minima Benchmark)         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Meta-Control Layer (`automl_engine/engine`)

* **`DynamicOptimizer`:** Coordinates the multi-step lifecycle, from initial random sampling to tournament evaluation and Tier 2 exploitation loops.
* **`UCB1Bandit`:** Implements the Upper Confidence Bound algorithm:
  $$UCB1_i = \bar{X}_i + c \cdot \sqrt{\frac{\ln N}{n_i}}$$
  Where $\bar{X}_i$ is the composite score, $N$ is total evaluations used, and $n_i$ is evaluations granted to arm $i$.
* **`CompositeScoringEngine`:** Evaluates every algorithm island across 4 normalized metrics:
  $$\text{Composite} = w_1 \cdot \text{Fitness} + w_2 \cdot \text{Speed} + w_3 \cdot \text{Diversity} + w_4 \cdot \text{Trend}$$
* **`TierManager`:** Manages the migration of elite solutions and stagnation triggers between Tier 1 (40% population) and Tier 2 (60% population).

---

## 2. Worker Optimizer Layer (`automl_engine/optimizers`)

All optimizers conform to a decoupled **Ask / Tell** interface:
* `ask(n)`: Generates $n$ candidate latent vectors $\mathbf{z} \in [0, 1]^d$.
* `tell(individuals)`: Ingests evaluated candidate solutions with their fitness scores to update internal populations, velocity vectors, or covariance matrices.

---

## 3. Evaluator & Search Space Layer (`automl_engine/core`)

* **Universal Latent Space:** Maps mixed categorical, integer, and log-scale continuous parameters into a unified $[0, 1]^d$ hypercube.
* **Asynchronous Parallelism:** Evaluates candidate configurations across multi-core workers via `concurrent.futures.ThreadPoolExecutor`.
* **Early Stopping:** Monitors tree growth and aborts unpromising candidates after 10 stagnant rounds.
