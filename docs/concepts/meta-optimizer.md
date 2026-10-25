# Meta-Optimizer (UCB1 Multi-Armed Bandit)

## The Meta-Optimization Concept

In standard optimization, an algorithm directly modifies candidate solutions:

$$\mathbf{x}_{t+1} = \text{Algorithm}(\mathbf{x}_t, \nabla f, \dots)$$

In **EvoSynth**, the system operates at a higher abstraction level as a **Meta-Optimizer**:

$$\text{Strategy}_{t+1} = \text{UCB1}(\text{Performance}(\mathcal{A}_1, \dots, \mathcal{A}_k))$$

The meta-optimizer treats each candidate search algorithm as an "arm" in a Multi-Armed Bandit framework. Instead of committing the entire evaluation budget (e.g., 2,500 model training runs) to one algorithm, EvoSynth adaptively routes budget chunks to whichever algorithm is demonstrating superior progress on the current landscape.

---

## UCB1 Formulation

The Upper Confidence Bound (UCB1) algorithm balances **Exploitation** (awarding trials to currently high-scoring algorithms) with **Exploration** (guaranteeing that stagnant or under-sampled algorithms are not permanently starved of budget):

$$\text{Score}_i = \bar{X}_i + c \cdot \sqrt{\frac{\ln N}{n_i}}$$

Where:
* $\bar{X}_i \in [0, 1]$ is the normalized **Composite Score** for algorithm $i$.
* $N$ is the total number of evaluations consumed across all algorithms.
* $n_i$ is the number of evaluations consumed specifically by algorithm $i$.
* $c$ is the exploration coefficient (default: $c = 0.1$).

### Why This Prevents Local Minima Trapping
If an aggressive optimizer like Particle Swarm Optimization (PSO) gets stuck in a local minimum, its composite improvement rate drops ($\bar{X}_{PSO} \to 0$). Simultaneously, the exploration bonus for Genetic Algorithm (GA) or Differential Evolution (DE) grows as $N$ increases. The UCB1 score forces the controller to divert resources to GA/DE, which break out of the local basin via stochastic crossover and differential mutations.
