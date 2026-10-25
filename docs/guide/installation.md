# Installation & Setup Guide

This guide covers setting up the Python optimization backend, dependencies, and the React telemetry dashboard.

---

## Prerequisites

* **Python:** Version 3.9+ (tested up to Python 3.14)
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

# Install backend dependencies
pip install -r requirements.txt
```

### Core Dependencies (`requirements.txt`)
* `ConfigSpace`: High-dimensional search space boundaries and categorical distributions.
* `numpy` & `scipy`: Vector operations and scientific computing.
* `xgboost` & `lightgbm`: Gradient boosted tree models.
* `scikit-learn`: Datasets, metrics, cross-validation, and Random Forest models.
* `torch` & `torchvision`: Deep learning framework for Convolutional Neural Network NAS.
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
python -c "from automl_engine.core.search_space import get_xgboost_search_space; from automl_engine.core.evaluator import XGBoostEvaluator; print('EvoSynth Backend Ready!')"
```
