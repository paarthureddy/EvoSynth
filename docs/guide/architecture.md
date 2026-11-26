# System Architecture

EvoSynth is structured into three clean, decoupled architectural layers:

```text
EvoSynth System Architecture
┌────────────────────────────────────────────────────────────────────────┐
│                        Meta-Control Layer                              │
│  • DynamicOptimizer (Controller)                                       │
│  • UCB1 Multi-Armed Bandit Controller                                  │
│  • Composite Scoring Engine (Fitness, Speed, Efficiency, Diversity,    │
│    Trend)                                                              │
│  • Tier Manager (stratified split, pairwise elite sharing)             │
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
│  • TabularEvaluator (XGBoost, LightGBM, RandomForest, ExtraTrees)      │
│  • ImageCNNEvaluator (PyTorch CNN Neural Architecture Search)          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Meta-Control Layer (`src/evosynth/engine`)

* **`DynamicOptimizer`:** Coordinates the run: initial population, stratified tier split, Tier 1 competition, promotion, Tier 2 checkpoints (SWAP, rollback, injection, elite sharing). It records every scoring call (`scoring_history`) and every decision (`decisions`).
* **`UCB1Bandit`:** Implements the Upper Confidence Bound algorithm:
  $$UCB1_i = \bar{X}_i + c \cdot \sqrt{\frac{\ln N}{n_i}}$$
  Where $\bar{X}_i$ is the composite score, $N$ is total evaluations used, $n_i$ is evaluations granted to arm $i$, and $c = 0.1$. Its ranking decides promotion and SWAP.
* **`CompositeScoringEngine`:** Scores every island on 5 min-max normalised metrics (see [Composite Scoring](../concepts/composite-scoring.md)):
  $$\text{Composite} = 0.30\,\text{Fitness} + 0.25\,\text{Speed} + 0.15\,\text{Efficiency} + 0.15\,\text{Diversity} + 0.15\,\text{Trend}$$
* **`TierManager`:** Splits the initial population (40% to four Tier 1 islands, 60% to two reserve groups, stratified by fitness rank) and performs pairwise elite sharing between the two Tier 2 islands.

---

## 2. Worker Optimizer Layer (`src/evosynth/optimizers`)

All optimizers conform to a decoupled **Ask / Tell** interface:
* `ask(n)`: Generates $n$ candidate latent vectors $\mathbf{z} \in [0, 1]^d$.
* `tell(individuals)`: Ingests evaluated candidate solutions with their fitness scores to update internal populations, velocity vectors, or covariance matrices.
* `inject(individuals)`: Replaces the worst members with already-evaluated individuals (random injection, elite sharing) and keeps the optimizer's internal state consistent.

Each optimizer draws all randomness from its own seeded generator, so a run is reproducible from `run.seed`.

---

## 3. Evaluator & Search Space Layer (`src/evosynth/core`, `evaluators`, `search_spaces`)

* **Universal Latent Space:** Maps mixed categorical, integer, and log-scale continuous parameters into a unified $[0, 1]^d$ hypercube.
* **Asynchronous Parallelism:** Evaluates candidate configurations across multi-core workers via `concurrent.futures.ThreadPoolExecutor`.
* **Early Stopping:** XGBoost and LightGBM stop after 10 rounds without improvement on a slice of the *training* fold; the scored fold is never used for early stopping.
* **Fixed splits:** Every dataset's held-out test rows are split off first; the search cross-validates on search rows, and the selected configuration is refitted on train rows and scored once on the test rows.
