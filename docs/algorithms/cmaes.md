# Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

## Overview

CMA-ES is widely regarded as the gold standard for difficult, continuous black-box optimization problems.

* **Implementation:** [`automl_engine/optimizers/cmaes.py`](file:///c:/Users/siddh/Desktop/College/EvoSynth/automl_engine/optimizers/cmaes.py)
* **Primary Role:** Second-order curvature learning and covariance estimation across non-convex continuous hyperparameter landscapes.

---

## Mathematical Formulation

CMA-ES samples candidate search points from a multivariate normal distribution:

$$\mathbf{x}_k^{(g+1)} \sim \mathbf{m}^{(g)} + \sigma^{(g)} \mathcal{N}\left(\mathbf{0}, \mathbf{C}^{(g)}\right) \quad \text{for } k = 1, \dots, \lambda$$

Where:
* $\mathbf{m}^{(g)}$ is the mean vector of the distribution at generation $g$.
* $\sigma^{(g)}$ is the step-size coordinate.
* $\mathbf{C}^{(g)}$ is the $d \times d$ covariance matrix describing pairwise parameter correlations.

### Updates:
After evaluating the $\lambda$ offspring, CMA-ES updates:
1. **Mean Vector:** Weighted recombination of the $\mu$ best individuals.
2. **Evolution Path:** Tracks historical step directions to accelerate progress.
3. **Covariance Matrix:** Gradually rotates and scales to match the contour lines of the objective function (approximating the inverse Hessian matrix without calculating derivatives).

---

## When CMA-ES Dominates

CMA-ES provides exceptional convergence when fine-tuning deep learning learning rates, weight decays, and continuous regularization terms once the primary model architecture is established.
