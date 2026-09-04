# EvoSynth: Dynamic Multi-Armed Bandit Meta-Optimizer for AutoML

## Overview

EvoSynth is an advanced Automated Machine Learning (AutoML) framework designed to dynamically optimize hyperparameter tuning processes. Unlike traditional AutoML systems that rely on a single static algorithm, EvoSynth operates as a meta-optimizer. It deploys multiple evolutionary algorithms simultaneously (e.g., Genetic Algorithms, Particle Swarm Optimization) and utilizes a Multi-Armed Bandit (UCB1) controller to intelligently allocate computational budget to the best-performing strategy in real-time.

## Key Innovations

- **Universal Latent Space Mapping:** Bypasses the limitation of applying mathematical optimizers to discrete or categorical data. All hyperparameters are mapped to a continuous n-dimensional space [0, 1], allowing swarm algorithms to compute velocities efficiently across mixed data types.
- **Dynamic Meta-Control (UCB1):** Balances "Exploitation" of currently successful algorithms with "Exploration" of stagnant ones. Strategies are continuously re-evaluated based on a composite score of fitness, speed, efficiency, diversity, and trend.
- **Cooperative Co-evolution (Island Model):** Mitigates genetic stagnation by structurally dividing populations into Active (Tier 1) and Background (Tier 2) islands, swapping elite individuals periodically to force global trajectory recalculation.
- **Real-Time Telemetry Dashboard:** A high-performance, React-based dashboard utilizing Server-Sent Events (SSE) to provide "glass-box" observability into the Multi-Armed Bandit's decision-making process.

## Architecture

![System Architecture](./assets/Architecture.png)

The system is highly modularized into three core layers:

1.  **Core Evaluator (`automl_engine/core`):** Handles problem definitions, search space bounds, and decodes the latent vectors to train machine learning models via k-fold cross-validation.
2.  **Meta-Engine (`automl_engine/engine`):** Houses the `DynamicOptimizer`, UCB1 Bandit, Tier Manager, and Composite Scoring engines.
3.  **Worker Optimizers (`automl_engine/optimizers`):** Contains the decoupled ask/tell implementations of the Genetic Algorithm and Particle Swarm Optimization.

## Directory Structure

```text
EvoSynth/
├── api_server.py                 # FastAPI Server-Sent Events (SSE) endpoint
├── main.py                       # CLI execution entry point
├── requirements.txt              # Python backend dependencies
├── automl_engine/                # The core AutoML engine
│   ├── core/                     # Evaluators, Individuals, and Search Space maps
│   ├── engine/                   # UCB1 Controller, Tier Manager, Composite Scorer
│   └── optimizers/               # Base algorithms (GA, PSO)
└── dashboard/                    # Live Telemetry UI (React + Vite + Tailwind CSS)
    ├── src/                      # Frontend source code
    └── package.json              # Node dependencies
```

## Installation & Setup

### 1. Python Backend (Engine)

Requires Python 3.9+.

```bash
# Navigate to the root directory
cd EvoSynth

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Live Telemetry Dashboard

Requires Node.js 18+.

```bash
# Navigate to the dashboard directory
cd dashboard

# Install dependencies
npm install
```

## Usage

To run the full visual EvoSynth suite, you must start both the backend API and the frontend dashboard.

### Start the Dashboard

```bash
cd dashboard
npm run dev
```

The dashboard will be available in your browser (default: `http://localhost:5173`).

### Start the AutoML Engine

Open a new terminal window in the project root directory.

```bash
# Ensure your virtual environment is active
python api_server.py
```

Once the Python server starts, it will immediately begin the optimization loop. You can monitor the live strategy switching, UCB1 calculations, accuracy improvements, and terminal output directly via the React dashboard.

## Future Roadmap

- Integration of Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
- Asynchronous distributed worker evaluations.
- Dynamic automation of the cooperative co-evolution island swapping.

## License

Proprietary / Academic Project. All rights reserved.
