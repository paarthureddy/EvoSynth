# Interpreting Telemetry Data

This guide explains how to read and interpret the real-time telemetry metrics displayed on the EvoSynth dashboard.

---

## 1. Diversity Metric ($D$)

* **Normal Range:** $0.20 - 0.70$
* **What it Means:** Population diversity measures how widely distributed candidate solutions are in the search space.
* **Low Diversity ($< 0.10$):** Indicates that the algorithm has collapsed onto a single local basin. If fitness is low, the controller triggers **Random Latent Injection** to force diversity back up.
* **High Diversity ($> 0.80$):** Indicates broad global exploration. Desirable in Tier 1 tournaments, but should decrease during Tier 2 exploitation.

---

## 2. Speed / Velocity Metric ($S$)

* **What it Means:** Measures fitness change across iterations ($\Delta \text{Accuracy}$).
* **Interpretation:**
  * High speed indicates an active gradient basin.
  * Zero speed indicates a plateau or convergence.

---

## 3. UCB Bonus vs. Composite Score

The total ranking score for any algorithm is:

$$\text{UCB Score} = \text{Composite Score} + \text{Bonus}$$

* **Composite Score:** Represents empirical performance (fitness, speed, diversity).
* **Bonus:** Represents the exploration premium for algorithms that have received fewer evaluation evaluations. If a high-potential algorithm was neglected due to initial randomness, the bonus increases logarithmically until the algorithm is granted another trial.
