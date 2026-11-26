# Particle Swarm Optimization (PSO)

## Overview

Particle Swarm Optimization mimics flocking birds or schooling fish. Each particle maintains a position vector $\mathbf{x}_i \in [0, 1]^d$ and a velocity vector $\mathbf{v}_i \in \mathbb{R}^d$.

* **Implementation:** [`src/evosynth/optimizers/pso.py`](https://github.com/paarthureddy/EvoSynth/blob/main/src/evosynth/optimizers/pso.py) (constants in [`settings/optimizers.toml`](https://github.com/paarthureddy/EvoSynth/blob/main/settings/optimizers.toml))
* **Primary Role:** Aggressive exploitation and high-speed gradient descent towards promising attractor basins.

---

## Velocity & Position Update Equations

At each iteration, particle velocities and positions are updated according to:

$$\mathbf{v}_i^{t+1} = w \cdot \mathbf{v}_i^t + c_1 r_1 (\mathbf{p}_i - \mathbf{x}_i^t) + c_2 r_2 (\mathbf{g} - \mathbf{x}_i^t)$$

$$\mathbf{x}_i^{t+1} = \text{clip}(\mathbf{x}_i^t + \mathbf{v}_i^{t+1}, 0, 1)$$

Where:
* $w = 0.5$ is the inertia weight controlling momentum (EvoSynth default).
* $c_1 \approx 1.5$ is the cognitive coefficient pulling the particle towards its personal best $\mathbf{p}_i$.
* $c_2 \approx 1.5$ is the social coefficient pulling the particle towards the global swarm best $\mathbf{g}$.
* $r_1, r_2 \sim U(0, 1)$ are stochastic weights.

When PSO is seeded with an already-evaluated population, those members become the particles' starting positions and personal bests, and the first step already moves them, so no evaluation is spent re-scoring a known configuration. Particles are moved round-robin, so every particle moves even when the controller asks for fewer candidates than the swarm size. When members are injected (stagnation or elite sharing), the worst particles are reset to the injected positions.

---

## When PSO Dominates

PSO dominates in the middle-to-late optimization phases when continuous hyperparameter regions (such as learning rate and regularization) are ready for fine-tuning.
