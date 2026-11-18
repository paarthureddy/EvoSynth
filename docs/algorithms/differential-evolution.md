# Differential Evolution (DE)

## Overview

Differential Evolution is a continuous stochastic search heuristic that uses vector differences for perturbing the population.

* **Implementation:** [`automl_engine/optimizers/de.py`](file:///c:/Users/siddh/Desktop/College/EvoSynth/automl_engine/optimizers/de.py)
* **Primary Role:** Robust navigation across non-linear, non-separable, and correlated hyperparameter surfaces.

---

## Mutation & Crossover Mechanism

For each target vector $\mathbf{x}_i$, three distinct individuals $\mathbf{x}_{r1}, \mathbf{x}_{r2}, \mathbf{x}_{r3}$ are sampled:

1. **Mutant Vector Generation:**
   $$\mathbf{v}_i = \mathbf{x}_{r1} + F \cdot (\mathbf{x}_{r2} - \mathbf{x}_{r3})$$
   Where $F \in [0.5, 0.9]$ is the differential scaling factor.

2. **Binomial Crossover:**
   $$\mathbf{u}_{i, j} = \begin{cases} \mathbf{v}_{i, j} & \text{if } \text{rand}_j \le CR \text{ or } j = j_{\text{rand}} \\ \mathbf{x}_{i, j} & \text{otherwise} \end{cases}$$
   Where $CR \in [0.1, 0.9]$ is the crossover rate.

3. **Selection:**
   $$\mathbf{x}_i^{t+1} = \begin{cases} \mathbf{u}_i & \text{if } f(\mathbf{u}_i) \ge f(\mathbf{x}_i^t) \\ \mathbf{x}_i^t & \text{otherwise} \end{cases}$$

---

## When DE Dominates

DE shines on ill-conditioned or rotated objective landscapes where parameter interactions (e.g., interaction between `learning_rate` and `max_depth`) make axis-aligned coordinate search ineffective.
