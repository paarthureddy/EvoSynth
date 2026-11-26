# Genetic Algorithm (GA)

## Overview

The Genetic Algorithm component simulates biological natural selection using chromosomes of continuous latent vectors $\mathbf{z} \in [0, 1]^d$.

* **Implementation:** [`src/evosynth/optimizers/ga.py`](https://github.com/paarthureddy/EvoSynth/blob/main/src/evosynth/optimizers/ga.py) (constants in [`settings/optimizers.toml`](https://github.com/paarthureddy/EvoSynth/blob/main/settings/optimizers.toml))
* **Primary Role:** Global exploration and escaping deceptive basins through stochastic genetic operators.

---

## Key Operators

1. **Tournament Selection:**
   * Samples $k = 2$ individuals at random and selects the fitter one as a parent.
2. **Arithmetic Crossover:**
   * Each gene of the child is a random blend of the two parents: $z_j = \alpha_j p_{1,j} + (1 - \alpha_j) p_{2,j}$ with $\alpha_j \sim U(0, 1)$.
3. **Gaussian Mutation:**
   * Each gene is perturbed with probability 0.1: $z_j' = \text{clip}(z_j + \mathcal{N}(0, 0.1^2), 0, 1)$.
4. **Elitist Survival:**
   * Parents and offspring are pooled and the best `pop_size` survive, so the best individual is never lost.

---

## When GA Dominates

GA is the preferred strategy in early optimization phases on non-continuous, highly rugged, or discrete loss surfaces where velocity vectors from swarms become noisy or unstable.
