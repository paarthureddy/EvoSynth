# Composite Scoring Engine

## Beyond Raw Accuracy

A naive meta-optimizer would judge algorithms solely on their highest single accuracy value. However, this creates severe failure modes:
* An algorithm might hit an isolated, ungeneralizable spike while having a stagnant, collapsed population.
* An algorithm might be making rapid, consistent improvements from a lower baseline, but get discarded before reaching its peak.

To make robust meta-optimization decisions, EvoSynth employs a multi-dimensional **Composite Scoring Engine**.

---

## The Four Evaluation Pillars

Every algorithm island is continuously evaluated across 4 normalized metrics:

```text
                  Composite Score
                        │
    ┌───────────┬───────┴───────┬───────────┐
    ▼           ▼               ▼           ▼
 Fitness      Speed         Diversity     Trend
 (Accuracy) (Velocity)     (Spread/Std)  (Delta)
```

1. **Fitness ($F$):**
   $$\text{Fitness} = \frac{\max(\text{pop}) - \text{baseline}}{\text{global\_best} - \text{baseline} + \epsilon}$$
   Measures absolute peak accuracy achieved by the algorithm's best individual.

2. **Speed / Velocity ($S$):**
   $$S = \max(\text{Best}_t - \text{Best}_{t-1}, 0.0)$$
   Measures the rate of fitness improvement per iteration.

3. **Population Diversity ($D$):**
   $$D = \min\left( \frac{\sigma(\text{fitnesses})}{\text{spread} + \epsilon}, 1.0 \right)$$
   Standard deviation of fitness across the island divided by spread. High diversity indicates the algorithm is still exploring wide areas of the search space rather than prematurely converging.

4. **Convergence Trend ($T$):**
   Evaluates momentum across rolling checkpoint windows to distinguish between accelerating and plateaued searches.

---

## Formula Integration

The composite score is combined as:

$$\bar{X}_i = w_f \cdot F + w_s \cdot S + w_d \cdot D + w_t \cdot T$$

This balanced score is then fed directly into the **UCB1 Multi-Armed Bandit** formula, allowing EvoSynth to reward algorithms that balance high accuracy with healthy diversity and forward momentum.
