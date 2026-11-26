# Differential Evolution (DE)

## Overview

Differential Evolution is a continuous stochastic search heuristic that uses vector differences for perturbing the population.

* **Implementation:** [`src/evosynth/optimizers/de.py`](https://github.com/paarthureddy/EvoSynth/blob/main/src/evosynth/optimizers/de.py) (constants in [`settings/optimizers.toml`](https://github.com/paarthureddy/EvoSynth/blob/main/settings/optimizers.toml))
* **Primary Role:** Robust navigation across non-linear, non-separable, and correlated hyperparameter surfaces.

---

## Mutation & Crossover Mechanism

For each target vector $\mathbf{x}_i$, three distinct individuals $\mathbf{x}_{r1}, \mathbf{x}_{r2}, \mathbf{x}_{r3}$ are sampled:

1. **Mutant Vector Generation:**
   $$\mathbf{v}_i = \mathbf{x}_{r1} + F \cdot (\mathbf{x}_{r2} - \mathbf{x}_{r3})$$
   Where $F$ is the differential scaling factor (EvoSynth default $F = 0.8$).

2. **Binomial Crossover:**
   $$\mathbf{u}_{i, j} = \begin{cases} \mathbf{v}_{i, j} & \text{if } \text{rand}_j \le CR \text{ or } j = j_{\text{rand}} \\ \mathbf{x}_{i, j} & \text{otherwise} \end{cases}$$
   Where $CR$ is the crossover rate (EvoSynth default $CR = 0.7$).

3. **Selection (one-to-one):**
   $$\mathbf{x}_i^{t+1} = \begin{cases} \mathbf{u}_i & \text{if } f(\mathbf{u}_i) \ge f(\mathbf{x}_i^t) \\ \mathbf{x}_i^t & \text{otherwise} \end{cases}$$
   Each trial competes only with the target it was built from. Targets are visited round-robin, so when the controller asks for fewer trials than the population size, successive asks still cover every member.

---

## When DE Dominates

DE shines on ill-conditioned or rotated objective landscapes where parameter interactions (e.g., interaction between `learning_rate` and `max_depth`) make axis-aligned coordinate search ineffective.
