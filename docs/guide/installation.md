# Installation & Setup Guide

This guide covers setting up the Python optimization backend, dependencies, and the React telemetry dashboard.

---

## Prerequisites

* **Python:** Version 3.11+ (developed and tested on Python 3.14)
* **Node.js:** Version 18+ (for the dashboard)
* **Package Managers:** `pip` and `npm`

---

## 1. Backend Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/paarthureddy/EvoSynth.git
cd EvoSynth

# Create virtual environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Linux/macOS:
source venv/bin/activate

# Install EvoSynth and its dependencies (pinned in pyproject.toml)
pip install -e ".[all]"        # or just `pip install -e .` without NAS / baseline extras
```

This installs the `evosynth` command (also runnable as `python -m evosynth`).

### Core Dependencies (`pyproject.toml`)
* `ConfigSpace`: High-dimensional search space boundaries and categorical distributions.
* `numpy` & `scipy`: Vector operations and scientific computing.
* `xgboost` & `lightgbm`: Gradient boosted tree models.
* `scikit-learn`: Datasets, metrics, cross-validation, and Random Forest models.
* `torch` & `torchvision` (extra `nas`): Deep learning framework for Convolutional Neural Network NAS.
* `optuna` & `flaml` (extra `baselines`): competitor baselines.
* `fastapi` & `uvicorn`: High-throughput Server-Sent Events (SSE) telemetry backend.
* `cma`: Covariance Matrix Adaptation Evolution Strategy.

---

## 2. Dashboard Setup

Open a new terminal window and navigate to the `dashboard` directory:

```bash
cd dashboard
npm install
npm run dev
```

The Vite dev server will host the frontend at `http://localhost:5173`.

---

## 3. Verifying Installation

Run the quick verification command:

```bash
evosynth settings check      # validates the settings/ folder
evosynth datasets list       # registered datasets
evosynth run                 # quick search + refit (preset search_demo)
```
