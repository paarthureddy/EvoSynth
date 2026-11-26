# Universal Latent Space Continuous Mapping

## The Continuous-Discrete Dilemma

Advanced nature-inspired optimizers—specifically **Particle Swarm Optimization (PSO)** and **CMA-ES**—are mathematically defined on continuous vector spaces:

$$v_{i}^{t+1} = w v_i^t + c_1 r_1 (p_i - x_i) + c_2 r_2 (g - x_i)$$

However, real-world Machine Learning hyperparameter tuning involves mixed data types:
* **Continuous:** Learning rate $\in [0.01, 0.3]$ (log scale), subsample $\in [0.5, 1.0]$, colsample_bytree $\in [0.3, 1.0]$, reg_alpha $\in [0, 10]$.
* **Discrete / Integer:** Tree depth $\in [3, 10]$, number of estimators $\in [50, 500]$.
* **Categorical / Architectural:** Model family $\in \{\text{XGBoost}, \text{RandomForest}, \text{LightGBM}, \text{ExtraTrees}\}$, feature-engineering strategy $\in \{\text{none}, \text{stats}, \text{interactions}\}$.

Together these form EvoSynth's 8-dimensional tabular search space (2 categorical genes + 6 hyperparameters; ranges in `settings/search_spaces.toml`).

Standard implementations either crash when passing strings into velocity equations or resort to clunky integer rounding that breaks swarm inertia.

---

## The EvoSynth Solution: $[0, 1]^d$ Latent Projection

EvoSynth maps all hyperparameter dimensions into a unified, continuous $d$-dimensional unit hypercube:

$$\mathbf{z} \in [0, 1]^d$$

### Mapping Mechanics

1. **Continuous Hyperparameters:**
   * Linearly or logarithmically scaled to $[0, 1]$.
   * $\text{val} = \text{lower} + z \cdot (\text{upper} - \text{lower})$ (or logarithmic equivalent).

2. **Categorical & Model Architecture Choices:**
   * Given $k$ choices $\{C_0, C_1, \dots, C_{k-1}\}$, the unit interval $[0, 1]$ is partitioned into $k$ equal bins:
     $$\text{Choice Index} = \min\left(\lfloor z \cdot k \rfloor, k - 1\right)$$
   * For example, for the 4 model families:
     * $z \in [0.00, 0.25) \implies \text{XGBoost}$
     * $z \in [0.25, 0.50) \implies \text{RandomForest}$
     * $z \in [0.50, 0.75) \implies \text{LightGBM}$
     * $z \in [0.75, 1.00] \implies \text{ExtraTrees}$

3. **Advantages:**
   * Swarms compute velocities across all dimensions without boundary violations.
   * Crossover and mutation operators in GA and DE operate naturally on floating-point chromosomes.
   * Model selection is explored organically alongside continuous hyperparameters.
