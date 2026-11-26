# Composite Scoring Engine

## Beyond Raw Accuracy

A naive meta-optimizer would judge algorithms only by their best accuracy so far. That fails in two ways:
* An algorithm can hold one lucky high score while its population has collapsed and stopped improving.
* An algorithm can be improving quickly from a lower starting point and be discarded before it catches up.

EvoSynth therefore scores every island with five metrics and combines them into one **composite score**, which the [UCB1 bandit](./meta-optimizer.md) then uses for every promotion and SWAP decision.

---

## The Five Metrics

The metrics are measured at every scoring call (after Tier 1, and at every Tier 2 checkpoint), per island:

| Metric | Definition | What it rewards |
| :--- | :--- | :--- |
| **Fitness** $F$ | Best score in the island (the dataset's metric, in %) | Being good now |
| **Speed** $S$ | $\max(\text{best}_t - \text{best}_{t-1},\ 0)$: improvement of the family's best since its last scoring call | Making progress |
| **Efficiency** $E$ | $S$ divided by the seconds of evaluation time spent on that family since its last scoring call | Progress per unit of compute |
| **Diversity** $D$ | Mean pairwise Euclidean distance between the island's latent vectors, divided by $\sqrt{d}$ (so $D \in [0, 1]$) | Still exploring, not collapsed |
| **Trend** $T$ | Least-squares slope of the family's best over its last 3 scoring calls | Sustained momentum |

Diversity is measured in the search space itself, on the configurations, not on their scores: two configurations with the same accuracy but very different hyperparameters count as diverse.

---

## Normalisation and Weights

Each metric is min-max normalised across the islands scored together, so every metric lies in $[0, 1]$ and the best island on a metric gets 1. When all islands tie on a metric, each gets 0.5. The composite is a weighted sum:

$$\text{composite}_i = 0.30\,F_i + 0.25\,S_i + 0.15\,E_i + 0.15\,D_i + 0.15\,T_i$$

The weights live in `settings/engine.toml` (`[composite]`) and must sum to 1. Because the score is relative to the islands scored at the same moment, it compares algorithms against each other; it is not an absolute quality measure.

The composite then enters the [UCB1 formula](./meta-optimizer.md#ucb1-formulation), which adds an exploration bonus for families that have used fewer evaluations.
