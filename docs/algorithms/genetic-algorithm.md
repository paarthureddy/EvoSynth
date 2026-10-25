# Genetic Algorithm (GA)

## Overview

The Genetic Algorithm component simulates biological natural selection using chromosomes of continuous latent vectors $\mathbf{z} \in [0, 1]^d$.

* **Implementation:** [`automl_engine/optimizers/ga.py`](file:///c:/Users/siddh/Desktop/College/EvoSynth/automl_engine/optimizers/ga.py)
* **Primary Role:** Global exploration and escaping deceptive basins through stochastic genetic operators.

---

## Key Operators

1. **Tournament Selection:**
   * Samples $k$ individuals at random and selects the highest fitness individual as a parent.
2. **Simulated Binary Crossover (SBX):**
   * Combines parental chromosomes to create novel offspring vectors with adjustable spread.
3. **Gaussian Mutation:**
   * Applies perturbation: $z_i' = \text{clip}(z_i + \mathcal{N}(0, \sigma^2), 0, 1)$ to maintain genetic variation across generations.
4. **Elitism:**
   * Guarantees that the top-performing individuals survive unconditionally into subsequent generations.

---

## When GA Dominates

GA is the preferred strategy in early optimization phases on non-continuous, highly rugged, or discrete loss surfaces where velocity vectors from swarms become noisy or unstable.
