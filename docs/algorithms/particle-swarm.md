# Particle Swarm Optimization (PSO)

## Overview

Particle Swarm Optimization mimics flocking birds or schooling fish. Each particle maintains a position vector $\mathbf{x}_i \in [0, 1]^d$ and a velocity vector $\mathbf{v}_i \in \mathbb{R}^d$.

* **Implementation:** [`automl_engine/optimizers/pso.py`](file:///c:/Users/siddh/Desktop/College/EvoSynth/automl_engine/optimizers/pso.py)
* **Primary Role:** Aggressive exploitation and high-speed gradient descent towards promising attractor basins.

---

## Velocity & Position Update Equations

At each iteration, particle velocities and positions are updated according to:

$$\mathbf{v}_i^{t+1} = w \cdot \mathbf{v}_i^t + c_1 r_1 (\mathbf{p}_i - \mathbf{x}_i^t) + c_2 r_2 (\mathbf{g} - \mathbf{x}_i^t)$$

$$\mathbf{x}_i^{t+1} = \text{clip}(\mathbf{x}_i^t + \mathbf{v}_i^{t+1}, 0, 1)$$

Where:
* $w \approx 0.7$ is the inertia weight controlling momentum.
* $c_1 \approx 1.5$ is the cognitive coefficient pulling the particle towards its personal best $\mathbf{p}_i$.
* $c_2 \approx 1.5$ is the social coefficient pulling the particle towards the global swarm best $\mathbf{g}$.
* $r_1, r_2 \sim U(0, 1)$ are stochastic weights.

---

## When PSO Dominates

PSO dominates in the middle-to-late optimization phases when continuous hyperparameter regions (such as learning rate and regularization) are ready for fine-tuning.
