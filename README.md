# EvoSynth: Dynamic Multi-Armed Bandit Meta-Optimizer for Automated Model Selection and Hyperparameter Tuning

<p align="center">
  <img src="./docs/public/logo.png" alt="EvoSynth Logo" width="220" />
</p>

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61dafb.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **EvoSynth** is a next-generation Automated Machine Learning (AutoML) and Neural Architecture Search (NAS) framework. Rather than relying on a single static optimizer or expensive post-hoc ensembles, EvoSynth functions as a **Dynamic Meta-Optimizer**: it deploys multiple evolutionary and swarm intelligence algorithms simultaneously and uses a **Multi-Armed Bandit (UCB1)** controller to dynamically allocate computational budget in real time to the most effective strategy.

---

## 🎯 Why This Title? (Workflow-to-Title Mapping)

When defending this project before an evaluation panel, every keyword in the title maps directly to a concrete phase of the system workflow:

```
[ EvoSynth ]
    │
    ├── 1. "Dynamic Multi-Armed Bandit" ──► UCB1 Controller routes compute budget on-the-fly based on exploration bonuses
    │
    ├── 2. "Meta-Optimizer" ──────────────► Operates on a portfolio of 4 competing optimizers (GA, PSO, DE, CMA-ES)
    │
    ├── 3. "Automated Model Selection" ───► Chooses optimal architecture (XGBoost, LightGBM, Random Forest, PyTorch CNN)
    │
    └── 4. "Hyperparameter Tuning" ───────► Discovers optimal continuous/discrete parameters via Early Stopping & Latent Space
```

| Title Component | Concrete Workflow Stage | Source Implementation |
| :--- | :--- | :--- |
| **Dynamic Multi-Armed Bandit** | Calculates real-time Upper Confidence Bounds ($UCB1 = \bar{X}_i + c \sqrt{\frac{\ln N}{n_i}}$) across composite fitness, velocity, diversity, and trend metrics. | [`src/evosynth/engine/bandit.py`](src/evosynth/engine/bandit.py) |
| **Meta-Optimizer** | Controls and coordinates multiple underlying optimization algorithms (Genetic Algorithm, Particle Swarm Optimization, Differential Evolution, CMA-ES) in a 2-Tier Island Model. | [`src/evosynth/engine/controller.py`](src/evosynth/engine/controller.py) |
| **Automated Model Selection** | Decodes latent vectors to dynamically switch model families (`XGBoost`, `LightGBM`, `RandomForest`, or PyTorch `CNN`) within the same search loop. | [`src/evosynth/search_spaces/`](src/evosynth/search_spaces/) |
| **Hyperparameter Tuning** | Evaluates continuous & discrete parameters in parallel with early stopping to prevent wasted compute on suboptimal branches. | [`src/evosynth/evaluators/tabular.py`](src/evosynth/evaluators/tabular.py) |

---

## 🔄 End-to-End System Workflow & Architecture

<p align="center">
  <img src="./docs/public/arch.png" alt="EvoSynth End-to-End System Architecture" width="100%" />
</p>

The EvoSynth workflow operates in **7 discrete, automated stages**:

1. **Step 0 — Problem Definition & Universal Latent Mapping**: Maps mixed discrete, categorical, and continuous hyperparameter spaces and neural architectures into a unified continuous hypercube $[0, 1]^d$.
2. **Step 1 — Parallel Population Initialization**: Generates and evaluates the initial population asynchronously across CPU worker threads with early stopping.
3. **Step 2 — Two-Tier Co-evolutionary Split**: Separates candidate individuals into **Tier 1 (40% seed tournament pool)** for rapid heuristic discovery and **Tier 2 (60% reserve pool)** for exploitation.
4. **Step 3 — Tier 1 Tournament**: Simultaneous execution of 4 competing metaheuristic algorithms (Genetic Algorithm, Particle Swarm Optimization, Differential Evolution, CMA-ES) over candidate vectors.
5. **Step 4 — Meta-Control & Dynamic Promotion**: Evaluates candidate fitness, velocity, diversity, and trend via a Composite Scoring Engine and routes computational resources via a **UCB1 Multi-Armed Bandit** with exploration bonuses to promote top strategies.
6. **Step 5 — Tier 2 Exploitation & Safeguards**: Dual-island cooperative co-evolution with top-20% elite migration and automated random latent injections when stagnation is detected.
7. **Step 6 — Production Output & Glass-Box Observability**: Delivers a **single standalone, production-ready model** with zero inference penalty, while streaming live telemetry via Server-Sent Events (SSE) to the interactive React dashboard.

<details>
<summary><b>Click to expand Text Flowchart (Mermaid)</b></summary>

```mermaid
flowchart TD
    subgraph Step0["Step 0: Problem Definition"]
        A[Search Space: Discrete, Categorical & Continuous] --> B[Universal Latent Space: Continuous Hypercube [0, 1]^n]
    end

    subgraph Step1["Step 1: Population Initialization"]
        B --> C[Evaluate Initial Population in Parallel across CPU Cores]
    end

    subgraph Step2["Step 2: Tiered Split"]
        C --> D1[Tier 1: 40% Exploration Seed Population]
        C --> D2[Tier 2: 60% Reserved Pool for Promotion]
    end

    subgraph Step3["Step 3: Tier 1 Tournament"]
        D1 --> E1[Genetic Algorithm]
        D1 --> E2[Particle Swarm Optimization]
        D1 --> E3[Differential Evolution]
        D1 --> E4[CMA-ES]
        E1 & E2 & E3 & E4 --> F[Evaluate Iterations in Parallel with Early Stopping]
    end

    subgraph Step4["Step 4: Meta-Control & Promotion"]
        F --> G[Composite Scoring Engine: Fitness + Speed + Diversity + Trend]
        G --> H[UCB1 Multi-Armed Bandit Exploration Bonus]
        H --> I[Promote Top-2 Strategies to Tier 2 Exploitation]
    end

    subgraph Step5["Step 5: Tier 2 Exploitation & Safeguards"]
        I --> J1[Cooperative Co-evolution Islands]
        D2 -. Seed Transfer .-> J1
        J1 --> K[Elite Pool Migration: Swap Top 20% Solutions]
        J1 --> L{Stagnation Detected?}
        L -- Yes --> M[Random Latent Injection: Replace Bottom 20%]
        L -- No --> N[Continue Search Loop]
    end

    subgraph Step6["Step 6: Output & Observability"]
        N --> O[Optimal Architecture & Hyperparameters Discovered]
        O --> P[Single Standalone Model Output: Zero Inference Penalty]
        G & H & J1 -. Live Telemetry SSE .-> Q[React Glass-Box Dashboard]
    end
```

</details>

---

## 💡 Core Innovations & Theoretical Defense

### 1. How EvoSynth Differs from Model Ensembling
A frequent question in AutoML is: *"Why not just train multiple models and ensemble them (e.g., stacking, blending, or voting)?"*
* **Inference Overhead:** Model ensembles combine multiple trained estimators at prediction time. Serving an ensemble in production multiplies latency, memory usage, and operational maintenance by the number of models in the ensemble.
* **Single Optimal Deliverable:** EvoSynth uses dynamic algorithm selection strictly in the **offline search and optimization phase**. The competing evolutionary algorithms work together to discover a **single, lean, optimal model configuration** (or neural architecture) that delivers peak accuracy with zero inference latency overhead.
* **Search Diversity vs. Predictive Aggregation:** Ensembles average model predictions to smooth variance. EvoSynth coordinates diverse search strategies to explore complex, multi-modal parameter spaces and locate global optima that individual optimizers fail to find.

### 2. Theoretical Superiority: The "No Free Lunch" Defense
According to the **No Free Lunch (NFL) Theorem for Optimization** (Wolpert & Macready, 1997), no single optimization algorithm can outperform all others across all problem landscapes:
* Algorithms with strong global exploration (e.g., Genetic Algorithms with crossover/mutation) excel on rugged landscapes but converge slowly on unimodal basins.
* Algorithms with aggressive exploitation and momentum (e.g., Particle Swarm Optimization, CMA-ES) converge rapidly on smooth convex topologies but become permanently trapped in local minima or noisy label regimes.

EvoSynth overcomes the NFL limitation by maintaining a **dynamic portfolio of complementary search algorithms**. By continuously evaluating fitness velocity, population diversity, and convergence trends, the UCB1 Bandit routes computational resources on the fly to the algorithm best suited for the current phase of the optimization landscape.

### 3. Direct Comparison: EvoSynth vs. Optuna vs. FLAML vs. H2O.ai

| Feature / Capability | Optuna | FLAML (Microsoft) | H2O.ai AutoML | **EvoSynth (Our System)** |
| :--- | :--- | :--- | :--- | :--- |
| **Search Paradigm** | Single Bayesian (TPE/CMA) | Cost-Frugal Optimization (CFO) | Random/Grid + Bayesian | **UCB1 Dynamic Multi-Armed Bandit** |
| **Algorithm Strategy** | Single static sampler | Single randomized local search | Static portfolio / sequential | **Cooperative Co-evolution (4-Optimizer Swarm)** |
| **Strategy Switching** | ❌ None (Fixed sampler) | ❌ None | ❌ None | **✅ Live Tier SWAPs ($>10\%$ UCB delta)** |
| **Diversity Safeguard** | ❌ None | ❌ None | ❌ None | **✅ Explicit Latent Space Diversity Tracking** |
| **Feature Engineering** | ❌ Manual | Limited heuristics | Fixed preprocessing | **✅ Searchable Dataset-Agnostic Gene** |
| **Production Deliverable** | Single model | Single model | Heavy Stacked Ensemble (10+ models) | **Single fast model (~2ms latency)** |
| **Observability** | Optuna Dashboard (Static polling) | Terminal logs | Web tables | **Real-Time "Glass-Box" SSE Telemetry Dashboard** |

*(For complete mathematical derivations and empirical comparisons, see [docs/research/comparative_analysis.md](docs/research/comparative_analysis.md).)*

---

## 🌲 Real-World Benchmark: UCI Forest Covertype (581,012 Samples)

To prove practical utility on large-scale real-world data, EvoSynth was evaluated on the **UCI Forest Covertype dataset** (581,012 samples, 54 features, 7 classes):

```text
+-------------------------------------------------------------------------------+
| EVO-SYNTH FINAL BENCHMARK SUMMARY (UCI Forest Covertype)                      |
+-------------------------------------------------------------------------------+
| Raw Untuned Baseline Accuracy  : 80.97% – 81.48%                             |
| Search Phase CV Accuracy       : 86.39% (Search Slice)                       |
| Generalization Test Accuracy   : 93.47% (60,000 Train / 20,000 Unseen Test)  |
| Relative Error Reduction       : 64.6% (19.0% -> 6.53% error)                |
| Winning Meta-Optimizer         : T2_PSO (Promoted via Dynamic SWAP)           |
| Optimal Model Architecture     : ExtraTrees (334 Trees, Depth 8, Colsample 0.43)|
| Optimal Feature Engineering    : 'interactions' (Dataset-Agnostic Pairwise)  |
| Total Search & Refit Time      : 9 minutes 44 seconds                         |
+-------------------------------------------------------------------------------+
```

### Key Empirical Findings:
1. **Escape from Premature Convergence:** In Step 4, Genetic Algorithm reached 86.21% but its population diversity decayed to 0.450. In Step 5, Particle Swarm in Tier 1 maintained high diversity (0.500) and climbed to 85.20% (UCB = 0.720 vs GA's 0.547). EvoSynth executed a **live SWAP event**, demoting GA and promoting PSO, which went on to discover the optimal configuration.
2. **Dataset-Agnostic Feature Engineering:** Without using any column names, the system generated continuous pairwise differences ($c_i - c_j$) and Euclidean distances ($\sqrt{c_i^2 + c_j^2}$), boosting accuracy from 81.48% to 93.47%.
3. **Pareto-Optimal Production Cost:** Delivers ensemble-grade accuracy (93.47% $\rightarrow$ 96%+ on full dataset) with a **single deployable model** executing in ~2ms with a ~150MB RAM footprint, avoiding the latency and hosting costs of 10-model production ensembles.

---

## ⚡ Performance & Speed Optimizations

EvoSynth addresses the computational bottleneck of high-budget evolutionary searches (2500+ evaluations) through two architectural optimizations:

1. **Asynchronous Parallel Worker Architecture:**
   * Utilizes `concurrent.futures.ThreadPoolExecutor` to dispatch population vector evaluations across all available CPU cores simultaneously.
   * Decreased initial population warm-up time from ~45 seconds down to **~4 seconds** (over **10x speedup**).
2. **Early Stopping on Tree-Based Models:**
   * Aborts training of gradient boosted models (`XGBoost`, `LightGBM`) if validation loss fails to improve after 10 consecutive iterations.
   * Unpromising candidate hyperparameters are pruned rapidly, preserving compute for competitive regions.

---

## 📂 Repository Architecture & Directory Structure

```text
EvoSynth/
├── pyproject.toml                # Package metadata, pinned dependencies, `evosynth` command
├── requirements.txt              # `-e .[all]` (installs the package with every extra)
├── settings/                     # EVERY tunable value (TOML), one file per topic + README index
│   ├── engine.toml               # Budget, tiers, UCB c, composite weights, safeguards
│   ├── optimizers.toml           # GA / PSO / DE / CMA-ES constants
│   ├── search_spaces.toml        # Model families, FE strategies, hyperparameter ranges
│   ├── evaluation.toml           # CV, early stopping, feature engineering, refit
│   ├── datasets.toml             # One section per dataset
│   ├── experiments.toml          # Presets for each command
│   └── run.toml, telemetry.toml
├── src/evosynth/                 # The Python package
│   ├── cli.py                    # `evosynth run | scale | ablation | baselines | nas | serve | settings | datasets`
│   ├── config.py                 # Loads, merges and validates settings/
│   ├── core/                     # Individual, latent-vector decoding
│   ├── search_spaces/            # Tabular (9D), NAS and toy ConfigSpace spaces
│   ├── datasets/                 # Dataset registry, loaders (download to .data/), splits
│   ├── features/                 # Dataset-agnostic feature engineering
│   ├── models/                   # Model factory, post-search refit, CNN
│   ├── evaluators/               # TabularEvaluator, ImageCNNEvaluator, toy objective
│   ├── engine/                   # DynamicOptimizer, UCB1 bandit, composite scoring, tiers
│   ├── optimizers/               # GA, PSO, DE, CMA-ES (ask / tell)
│   ├── telemetry/                # Event reporter, run logging, FastAPI SSE server
│   └── experiments/              # search, scaling, ablation, baselines, nas, run folders
├── results/                      # One folder per run: <experiment>/<dataset>/<variant>/seed_<k>/
│   └── archive/                  # Panel-review (v1) results, kept unchanged
├── tools/run_experiments.ps1     # Staged runner for the full Covertype study
├── dashboard/                    # React live telemetry dashboard (Vite + Tailwind CSS)
├── docs/                         # VitePress documentation site
│   ├── research/                 # Comparative analysis, related papers
│   └── public/                   # Logo and architecture figure
├── handbook/                     # Internal project handbook + panel-defense guide (not on the site)
└── paper/                        # Paper and slide sources (LaTeX), figures, links
```

---

## 🚀 Quickstart & Usage

### 1. Installation

Requires Python 3.11+ (developed on 3.14).

```bash
# Clone the repository
git clone https://github.com/paarthureddy/EvoSynth.git
cd EvoSynth

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate # On Linux/macOS

# Install EvoSynth with all extras (NAS, baselines, tests)
pip install -e ".[all]"
```

This installs the `evosynth` command (also available as `python -m evosynth`). Datasets are
downloaded on first use into `.data/` (git-ignored), or all at once with `evosynth datasets download`
(about 150 MB). `evosynth datasets list` shows every registered dataset, its metric and whether it is cached.

### 2. Run Real-World Benchmark (UCI Forest Covertype)
Executes dynamic meta-optimization across 4 model families (XGBoost, LightGBM, RandomForest, ExtraTrees) with automated dataset-agnostic feature engineering and post-search generalization refit:
```bash
evosynth run                         # Fast benchmark (preset search_demo: 200 evals, pop=40)
evosynth run --full                  # Full benchmark (preset search_full: 500 evals, pop=40)
evosynth scale --budget 500          # Budget-scaling run with checkpoints (300 / 500 / 1000)
evosynth ablation --setup CVGS-500   # Isolated GA / PSO / DE / CMA-ES vs EvoSynth
evosynth baselines --test all        # Default ExtraTrees, Random Search, Optuna, FLAML
evosynth run --dataset higgs         # any registered dataset: covertype (default), higgs, adult,
                                     # airlines, poker_hand, miniboone, jannis
```

Every method sees the same rows: a held-out test set is split off first, and the search and the
refit draw from the rest (`settings/datasets.toml`). Each dataset is scored with its own metric
(accuracy for the multiclass datasets, ROC AUC for the binary ones). After the search, the top 5
configurations are re-scored with 5-fold CV before the winner is chosen (`[selection]`).

Each run writes to `results/<experiment>/<dataset>/<variant>/seed_<k>/` (`manifest.json` with the
exact settings, git commit and library versions; `result.json`; `run.log`; ...).

### 3. Change Any Setting
All hyperparameters, weights and presets live in [`settings/`](settings/README.md). Edit a file, or
override a value for one run:
```bash
evosynth settings show engine
evosynth run --set bandit.c=0.2 --set composite.fitness=0.4 --set composite.speed=0.15
```

### 4. Run Deep Learning Vision NAS
Discovers optimal CNN layer topology (convolutional channels, fully connected units, activation functions) and training hyperparameters in PyTorch, on Fashion-MNIST (10,000 training / 2,500 validation images during search; the official test set is held out):
```bash
evosynth nas
```

### 5. Launch Live Telemetry Dashboard

In Terminal 1 (Start the Telemetry API):
```bash
evosynth serve
```

In Terminal 2 (Start the React Dashboard):
```bash
cd dashboard
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser to monitor algorithm promotion, real-time UCB1 scores, diversity trends, and telemetry logs live.

---

## 📜 Documentation Site

To browse the complete technical documentation site locally:
```bash
cd docs
npm install
npm run docs:dev
```
Open [http://localhost:5173](http://localhost:5173) (or the displayed VitePress port) to view full guides, algorithm math, API references, and developer tutorials.

The complete project handbook (architecture, engine internals, experiments, results, setup) is in
[`handbook/`](handbook/README.md); the paper and slides are in [`paper/`](paper/README.md).

---

## 📄 License & Attribution

Developed as an Academic / College Capstone Project. All rights reserved.
