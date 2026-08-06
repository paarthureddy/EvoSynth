# Hybrid AutoML Strategy Switching — Complete End-to-End Workflow
## Tiered Islands with Bandit-Based Promotion

Below is a fully simulated, end-to-end run with concrete values, decisions, and edge cases — matching the granularity of your original workflow.

---

## 🎯 Scenario Setup

| Parameter | Value |
|---|---|
| Task | CIFAR-10 classification (dataset agnostic — any supervised learning task) |
| Total Population | N = 50 |
| Strategies | GA, PSO, DE, CMA-ES |
| Tier 1 (Competition) | 40% → 20 individuals |
| Tier 2 (Working) | 60% → 30 individuals |
| Sharing interval (k) | Every 5 iterations |
| Top x% shared | 20% |
| Composite weights | Accuracy=0.30, Speed=0.25, Efficiency=0.15, Diversity=0.15, Trend=0.15 |

---

## 🔍 Search Space

The search space defines **all possible hyperparameters and architectural choices** that a candidate model (individual) can take. The population is generated from this space — **NOT from the dataset**. The dataset is only used later to evaluate (score) each individual.

### Full Search Space (for CIFAR-10 / Image Classification)

| Category | Parameter | Range / Options | Type |
|---|---|---|---|
| **Architecture** | Number of conv layers | 2–10 | Integer |
| | Filters per layer | 16, 32, 64, 128, 256 | Categorical / Integer |
| | Kernel size | 3×3, 5×5, 7×7 | Categorical |
| | Pooling type | Max, Average, None | Categorical |
| | Dense layer units | 64, 128, 256, 512 | Categorical |
| | Number of dense layers | 1–3 | Integer |
| | Activation function | ReLU, LeakyReLU, ELU, Swish | Categorical |
| | Use batch normalization | Yes / No | Boolean |
| | Dropout rate | 0.0–0.5 | Continuous |
| **Training** | Learning rate | 0.0001–0.01 (log-scale) | Continuous |
| | Optimizer | Adam, SGD, RMSprop, AdamW | Categorical |
| | Batch size | 16, 32, 64, 128 | Categorical |
| | Weight decay | 0.0–0.01 | Continuous |
| | LR scheduler | StepLR, CosineAnnealing, None | Categorical |

> [!NOTE]
> For this walkthrough, we use a **simplified 5-parameter version** for clarity. The system generalizes to the full search space above.

### Simplified Search Space (used in this walkthrough)

```
Layers:    3–10
Filters:   32–256
Kernel:    {3x3, 5x5}
LR:        0.0001–0.01
Optimizer: {Adam, SGD, RMSprop}
```

### Key Conceptual Points

| Concept | Explanation |
|---|---|
| **Population comes from search space** | Individuals are randomly sampled configurations — NOT derived from the dataset |
| **Dataset judges, not generates** | The dataset is the **evaluation environment** — it produces the fitness score for each individual |
| **Indirect dataset influence** | Over generations, evolution biases the population toward dataset-specific good solutions (via selection pressure) |
| **Larger search space = more exploration** | More parameters → more generations needed to find optimal configurations |

---

## 🧬 Representation

Each individual is a vector:
```
[Layers, Filters, Kernel, LR, Optimizer]
```
Example:
```
I1 = [4, 64, 3x3, 0.001, Adam]
```

---

## 🔷 PHASE 1: INITIALIZATION

### Step 1: Generate Population (N=50)

All 50 individuals are randomly sampled from the search space:

```
Layers:   3–10
Filters:  32–256
Kernel:   {3x3, 5x5}
LR:       0.0001–0.01
Optimizer: {Adam, SGD, RMSprop}
```

Each individual is evaluated on CIFAR-10 (quick 5-epoch training) to get initial fitness.

| ID Range | Count | Accuracy Range |
|---|---|---|
| I1–I20 | 20 | 72%–83% |
| I21–I50 | 30 | 70%–84% |

---

### Step 2: Tier Allocation

Sort all 50 by fitness. Assign randomly (NOT by fitness) to tiers — tiers must start with comparable distributions.

| Tier | Purpose | Size | Assignment |
|---|---|---|---|
| **Tier 1** (Competition) | 40% | 20 | Split among GA(5), PSO(5), DE(5), CMA-ES(5) |
| **Tier 2** (Working) | 60% | 30 | Temporarily unassigned — needs warm-up |

---

## 🔷 PHASE 2: TIER-1 WARM-UP (Iterations 1–5)

### Step 3: Each Strategy Evolves Its Tier-1 Batch Independently

| Island | Strategy | Individuals | Init Best | Init Avg |
|---|---|---|---|---|
| T1-GA | GA | I1, I5, I9, I13, I17 | 80% | 76.2% |
| T1-PSO | PSO | I2, I6, I10, I14, I18 | 79% | 75.8% |
| T1-DE | DE | I3, I7, I11, I15, I19 | 82% | 77.5% |
| T1-CMA | CMA-ES | I4, I8, I12, I16, I20 | 81% | 76.8% |

---

### Step 4: Apply Strategies (5 iterations on Tier 1 only)

**🔹 GA on its batch (5 individuals):**
- Crossover(I1, I5) → G1 = [5, 96, 3x3, 0.0008, Adam] → 82.5%
- Mutation(I9) → G2 = [4, 72, 3x3, 0.0012, Adam] → 81%
- Replace worst with G1, G2

After 5 iters, GA batch:
```
Best: 83.5%  |  Avg: 80.1%  |  Diversity: 0.68
```

**🔹 PSO on its batch (5 individuals):**
- Update velocities toward personal & global best (I6)
- P1 = [5, 70, 3x3, 0.002, SGD] → 81%
- Replace worst

After 5 iters, PSO batch:
```
Best: 81.2%  |  Avg: 79.0%  |  Diversity: 0.55
```

**🔹 DE on its batch (5 individuals):**
- Differential mutation + crossover
- D1 = [7, 200, 3x3, 0.0002, SGD] → 84.5%
- Replace worst

After 5 iters, DE batch:
```
Best: 84.5%  |  Avg: 81.3%  |  Diversity: 0.74
```

**🔹 CMA-ES on its batch (5 individuals):**
- Covariance matrix adaptation + sampling
- C1 = [6, 150, 3x3, 0.0003, Adam] → 83.8%
- Replace worst

After 5 iters, CMA-ES batch:
```
Best: 83.8%  |  Avg: 80.5%  |  Diversity: 0.70
```

---

### Step 5: Compute Composite Scores (End of Warm-Up)

**How Composite Scoring works:**

Each strategy is evaluated on 5 performance metrics: **Accuracy** (how good its best solution is), **Convergence Speed** (how fast it's improving per iteration), **Computational Efficiency** (how much fitness gain per unit of compute time), **Population Diversity** (how varied its solutions are — prevents premature convergence), and **Performance Trend** (whether it's improving, flat, or declining over recent iterations).

Each metric is **normalized** using min-max scaling across all 4 strategies, so scores range from 0 (worst among strategies) to 1 (best among strategies). The final composite score is a **weighted sum** of these normalized metrics:

**Normalization formula (per metric, per algorithm):**
```
M_norm(S) = ( M(S) - M_min ) / ( M_max - M_min )
```
Where `M(S)` is the raw metric value for algorithm S, and `M_min`/`M_max` are the min/max across all 4 algorithms for that metric.

**Composite Score formula:**
```
Score(S) = Σ (w_i × M_i_norm(S))   for i = 1 to 5

         = 0.30 × Accuracy_norm
         + 0.25 × Speed_norm
         + 0.15 × Efficiency_norm
         + 0.15 × Diversity_norm
         + 0.15 × Trend_norm
```

- Accuracy gets the highest weight (0.30) — the primary goal is finding good solutions
- Convergence Speed is next (0.25) — faster improvement means better use of budget
- Efficiency, Diversity, and Trend each get 0.15 — important but secondary

**Higher composite score = better-performing algorithm.** This score drives all promotion/demotion decisions.

| Metric | GA | PSO | DE | CMA-ES |
|---|---|---|---|---|
| Accuracy (best) | 83.5% | 81.2% | 84.5% | 83.8% |
| Convergence Speed (Δf/iter) | +0.70 | +0.44 | +0.90 | +0.76 |
| Comp. Efficiency (fitness/sec) | 0.85 | 0.90 | 0.78 | 0.72 |
| Diversity (σ) | 0.68 | 0.55 | 0.74 | 0.70 |
| Performance Trend (slope) | +0.65 | +0.40 | +0.85 | +0.72 |

**Weighted Composite (after min-max normalization):**
```
DE     = 0.800
CMA-ES = 0.680
GA     = 0.624
PSO    = 0.150
```

**Ranking:** DE (0.800) > CMA-ES (0.680) > GA (0.624) > PSO (0.150)

---

### Step 6: Assign Tier 2

Top 2 strategies → promoted to Tier 2 as islands.

| Tier | Strategy | Population Size |
|---|---|---|
| T1-GA | GA | 5 (continues in competition) |
| T1-PSO | PSO | 5 (continues in competition) |
| T1-DE | DE | 5 (continues in competition) |
| T1-CMA | CMA-ES | 5 (continues in competition) |
| **T2-Island-DE** | **DE** | **15** (ranked #1) |
| **T2-Island-CMA** | **CMA-ES** | **15** (ranked #2) |

The 30 Tier-2 individuals are split between the two promoted strategies.

👉 GA and PSO remain in Tier 1 only — they must earn their way up.

---

## 🔷 PHASE 3: PARALLEL OPERATION (Iterations 6–10)

### Step 7: All Tiers Evolve Simultaneously

**Tier 1** (all 4 strategies on their small batches — continuous competition):

| Island | Strategy | Size | Best After 5 More Iters |
|---|---|---|---|
| T1-GA | GA | 5 | 84.0% |
| T1-PSO | PSO | 5 | 82.5% |
| T1-DE | DE | 5 | 85.0% |
| T1-CMA | CMA-ES | 5 | 84.5% |

**Tier 2** (top 2 strategies on large batches — real work):

| Island | Strategy | Size | Best After 5 Iters | Avg |
|---|---|---|---|---|
| T2-DE | DE | 15 | 87.2% | 84.5% |
| T2-CMA | CMA-ES | 15 | 86.5% | 83.8% |

---

### Step 8: Tier-2 Island Sharing (After iteration 10, k=5)

Top 20% of each Tier-2 island = 3 individuals shared.

| Source | Shared Individuals | Accuracies |
|---|---|---|
| T2-DE | D_e1, D_e2, D_e3 | 87.2%, 86.8%, 86.5% |
| T2-CMA | C_e1, C_e2, C_e3 | 86.5%, 86.2%, 86.0% |

**Migration:**
- T2-DE receives CMA-ES's top 3 → replaces its worst 3
- T2-CMA receives DE's top 3 → replaces its worst 3

> After sharing, both islands have cross-pollinated genetic material.

---

## 🔷 PHASE 4: MONITORING & TIER-1 CHALLENGE

### Step 9: Tier-1 Composite Scores (After iteration 10)

| Metric | GA | PSO | DE | CMA-ES |
|---|---|---|---|---|
| Composite Score | 0.68 | 0.52 | 0.82 | 0.75 |

**Current Tier-2 Assignment:**
- T2-DE: Composite on Tier 2 = 0.78
- T2-CMA: Composite on Tier 2 = 0.71

---

### Step 10: Challenge Check

```
Rule: If any Tier-1 strategy's composite > any Tier-2 strategy's composite → Replace
```

Check:
- T1-GA (0.68) vs T2-CMA (0.71)? → 0.68 < 0.71 → ❌ No replacement
- T1-PSO (0.52) vs T2-CMA (0.71)? → 0.52 < 0.71 → ❌ No replacement
- T1-DE (0.82) vs T2-CMA (0.71)? → T1-DE is already in T2, skip
- T1-CMA (0.75) vs T2-CMA (0.71)? → Same strategy, skip

No T1 strategy beats a T2 strategy it's not already in → **No replacement this round.**

---

## 🔷 PHASE 5: CONTINUED EVOLUTION (Iterations 11–20)

### Step 11: Two More Full Cycles

**Tier 2 results after iteration 20:**

| Island | Strategy | Best | Avg | Composite |
|---|---|---|---|---|
| T2-DE | DE | 89.1% | 86.8% | 0.85 |
| T2-CMA | CMA-ES | 87.5% | 85.2% | 0.72 |

**Tier 1 results after iteration 20:**

| Island | Strategy | Best | Composite |
|---|---|---|---|
| T1-GA | GA | 85.2% | 0.71 |
| T1-PSO | PSO | 84.8% | 0.69 |
| T1-DE | DE | 86.0% | 0.83 |
| T1-CMA | CMA-ES | 85.5% | 0.74 |

---

### Step 12: Another Sharing Round (Iteration 20)

Tier-2 islands exchange top 20% again. Cross-pollination continues.

Post-sharing:
| Island | Best |
|---|---|
| T2-DE | 89.3% |
| T2-GA | 88.1% |

---

## 🔷 PHASE 6: CHALLENGE & PROMOTION

### Step 13: GA Improves in Tier 1

Suppose GA finds a strong crossover pattern at iteration 22:

| Island | Strategy | Composite |
|---|---|---|
| T1-GA | GA | **0.76** |
| T2-CMA | CMA-ES | **0.72** |

**Challenge:** GA (0.76) > CMA-ES at Tier 2 (0.72)

### Step 14: Promotion!

**Action:** GA replaces CMA-ES as a Tier-2 island operator.

| Before | After |
|---|---|
| T2-DE (15), T2-CMA (15) | T2-DE (15), **T2-GA (15)** |
| T1: GA(5), PSO(5), DE(5), CMA(5) | T1: GA(5), PSO(5), DE(5), **CMA(5)** |

**What happens to CMA-ES?**
- CMA-ES drops back to Tier 1 only (it already has its T1 batch of 5)
- CMA-ES can re-compete and earn promotion again later

**What happens to GA's Tier-2 batch?**
- GA inherits CMA-ES's 15 individuals
- GA applies its crossover/mutation operators to these individuals
- First 2 iterations are an adaptation period (no challenge checks during this time)

---

## 🔷 PHASE 7: STAGNATION HANDLING

### Step 15: Stagnation Detected at Tier 2

After iteration 30, T2-DE shows no improvement for 5 consecutive iterations:
```
Iter 26: Best = 89.5%
Iter 27: Best = 89.5%
Iter 28: Best = 89.6%
Iter 29: Best = 89.6%
Iter 30: Best = 89.6%
```

**Stagnation trigger for T2-DE = TRUE**

### Step 16: Bandit-Assisted Response

**How UCB1 Bandit Selection works:**

Instead of switching strategies blindly when stagnation is detected, the system uses **UCB1 (Upper Confidence Bound)** — a classic algorithm from the multi-armed bandit problem in reinforcement learning.

The UCB1 score for each strategy combines two parts:
1. **Exploitation (average past reward):** How well this strategy has performed historically. Strategies with higher average composite scores get a higher exploitation score.
2. **Exploration bonus:** A bonus that grows larger for strategies that have been used **fewer** times. This ensures that underused strategies — which might actually be great but haven't had enough chances — get selected occasionally.

The strategy with the **highest combined score** (exploitation + exploration bonus) is selected. This means:
- A well-tested strategy with consistent good results will be chosen (exploitation)
- BUT a rarely-used strategy can "jump ahead" due to its large exploration bonus
- Over time, all strategies get tested enough that the selection naturally gravitates toward the genuinely best one

The exploration constant `c` (typically √2 ≈ 1.414) controls this trade-off: higher c = more adventurous, lower c = more conservative.

The **UCB1 formula:**

```
UCB(S) = μ(S) + c × √( ln(N_total) / N(S) )
```

| Symbol | Meaning |
|---|---|
| **μ(S)** | Average reward of strategy S (mean composite score across all times it was used) |
| **c** | Exploration constant (typically c = √2 ≈ 1.414, tunable) |
| **N_total** | Total number of strategy evaluations across all strategies |
| **N(S)** | Number of times strategy S has been used/evaluated |
| **√(ln(N_total) / N(S))** | Exploration bonus — grows larger for underused strategies |

**Compute UCB for each strategy (N_total = 12):**

```
DE:     UCB = 1.8 + 1.414 × √(ln(12)/6) = 1.8 + 0.33 = 2.13
CMA-ES: UCB = 1.6 + 1.414 × √(ln(12)/3) = 1.6 + 0.59 = 2.19
GA:     UCB = 1.3 + 1.414 × √(ln(12)/2) = 1.3 + 0.88 = 2.18
PSO:    UCB = 1.1 + 1.414 × √(ln(12)/1) = 1.1 + 1.12 = 2.22  ← highest
```

| Strategy | Avg Reward (μ) | Times Used (N) | Exploration Bonus | UCB Score |
|---|---|---|---|---|
| DE | 1.8 | 6 | +0.33 | 2.13 |
| CMA-ES | 1.6 | 3 | +0.59 | 2.19 |
| GA | 1.3 | 2 | +0.88 | 2.18 |
| PSO | 1.1 | 1 | +1.12 | **2.22** |

**UCB selects PSO** — despite having the lowest average reward, PSO's high exploration bonus (used only 1 time) pushes it to the top. This ensures underused strategies get a fair chance.

> **Key trade-off:** High c → more exploration (try underused strategies). Low c → more exploitation (stick with what works). Tuning c lets you control how adventurous the system is.

### Step 17: Action on Stagnation

**Option chosen:** Diversity injection + strategy re-evaluation
- Replace bottom 20% of T2-DE's population (3 individuals) with random new individuals
- Give DE 3 more iterations to recover
- If still stagnant → trigger T1 challenge check immediately

After diversity injection:
```
Iter 31: Best = 89.6% (same — but new individuals exploring)
Iter 32: Best = 89.8% (new individual found promising region!)
Iter 33: Best = 90.1%
```

Stagnation resolved! DE continues.

---

### Step 17b: Tier-1 Stagnation Handling

Tier 1 also gets monitored. Two scenarios:

**Scenario A: Single T1 strategy stagnates**

Example: PSO shows no improvement for `w` consecutive scoring rounds.

| Round | PSO Best | PSO Composite | Trend |
|---|---|---|---|
| Round 4 | 82.5% | 0.52 | — |
| Round 5 | 82.5% | 0.50 | ↓ |
| Round 6 | 82.4% | 0.48 | ↓ |

**Action:**
```
1. Replace bottom 50% of PSO's batch (2–3 individuals) with fresh random individuals
2. Temporarily reduce PSO's batch from 5 → 3 individuals
3. Give the freed 2 individuals to the best-performing T1 strategy
4. After 2 scoring rounds, restore PSO's batch size if it recovers
```

**Scenario B: ALL T1 strategies stagnate simultaneously**

All composite scores plateau for `w` consecutive rounds — competition becomes meaningless.

| Round | GA | PSO | DE | CMA-ES |
|---|---|---|---|---|
| Round 6 | 0.65 | 0.50 | 0.72 | 0.68 |
| Round 7 | 0.65 | 0.50 | 0.71 | 0.67 |
| Round 8 | 0.64 | 0.49 | 0.71 | 0.67 |

**Action:**
```
Global T1 refresh:
1. Keep the single best individual from each T1 island (4 total)
2. Replace remaining 16 individuals with fresh random samples from search space
3. Re-run warm-up for 3 iterations before next scoring round
4. This gives all strategies fresh material to differentiate themselves
```

---

## 🔷 PHASE 8: TIER-2 SHARING WITH TIER-1 (Optional Enhancement)

### Step 18: Periodic Top-Down Knowledge Sharing

Every 10 iterations, the single best individual from each Tier-2 island is injected into ALL Tier-1 islands (replacing their worst):

| Round | Best from T2-DE | Best from T2-GA | Injected Into |
|---|---|---|---|
| Iter 20 | 89.3% | 88.1% | T1-GA, T1-PSO, T1-DE, T1-CMA |
| Iter 30 | 90.1% | 89.0% | T1-GA, T1-PSO, T1-DE, T1-CMA |

**Why?**
- T1 algorithms can test whether they can improve upon T2's best solutions
- If CMA-ES at T1 can further improve a 90.1% solution → strong signal that CMA-ES deserves promotion

---

## 🔷 PHASE 9: CONVERGENCE

### Step 19: System Approaches Final Solution

At iteration 40:

| Tier | Island | Best | Avg |
|---|---|---|---|
| T2-DE | DE | 91.2% | 89.5% |
| T2-GA | GA | 90.5% | 88.8% |
| T1-GA | GA | 88.0% | 86.5% |
| T1-PSO | PSO | 87.5% | 85.8% |
| T1-DE | DE | 88.8% | 87.0% |
| T1-CMA | CMA-ES | 88.2% | 86.8% |

### Step 20: Final Output

**Best individual overall:** From T2-DE
```
[8, 192, 3x3, 0.00015, SGD] → 91.2% accuracy on CIFAR-10
```

**Ensemble option:** Top 5 individuals across all tiers:
```
1. T2-DE:  91.2%
2. T2-DE:  90.8%
3. T2-GA:  90.5%
4. T2-DE:  90.3%
5. T2-GA:  90.1%
Ensemble prediction accuracy: 92.1% ← better than any individual!
```

---

## ⚠️ EDGE CASES

### ❌ Case 1: Tier-1 Batch Too Small to Evaluate Reliably

**Problem:** With many strategies and small Tier-1 allocation → batches too small → high variance.

**Solution:**
```
Run each T1 strategy for at least 5 iterations before computing composite
Use rolling average of composite scores (window = 3 rounds) instead of single-point
```

---

### ❌ Case 2: Promoted Strategy Fails on Larger Population

**Problem:** PSO was great on 3 individuals but collapses on 20 (can't handle diverse population).

**Solution:**
```
Enforce 5-iteration grace period after promotion
If no improvement after grace period → immediately demote and restore previous strategy
Keep a snapshot of the pre-promotion population state for rollback
```

---

### ❌ Case 3: All Tier-1 Strategies Perform Similarly

**Problem:** GA=0.65, PSO=0.63, DE=0.66, CMA-ES=0.64 → no clear winner.

**Solution:**
```
If max_score - min_score < threshold (e.g., 0.05):
    Don't promote any new strategy
    Keep current Tier-2 assignments stable
    Increase Tier-1 iteration count for more data
```

---

### ❌ Case 4: Oscillating Promotions (GA↔PSO↔GA↔PSO)

**Problem:** Two strategies keep replacing each other at Tier 2.

**Solution:**
```
Promotion cooldown: After a replacement, the new strategy must hold
position for at least 2k iterations before being challengeable
Require challenger to beat incumbent by margin δ (e.g., 0.05), not just exceed
```

---

### ❌ Case 5: Tier-2 Islands Converge to Same Region

**Problem:** After many sharing rounds, both T2 islands are exploring identical solution space.

**Solution:**
```
Track inter-island diversity (distance between island centroids)
If diversity < threshold:
    Reduce sharing frequency (k → 2k)
    Inject 10% random individuals into the less-fit island
    Temporarily pause sharing for 2 rounds
```

---

### ❌ Case 6: Full Population Collapse

**Problem:** All individuals across all tiers become very similar (diversity near 0).

**Solution:**
```
Soft reset:
    Keep top 30% of all individuals
    Replace remaining 70% with fresh random samples
    Reset all bandit statistics
    Re-run warm-up phase (Phase 2)
```

---

## 🔁 Important Design Choices

| Question | Answer | Reason |
|---|---|---|
| **Do we rollback on bad promotion?** | YES (after grace period) | Because promotion changes the algorithm operating on a large batch — failure is costly |
| **Do we reduce total population?** | NO | N always = 50, just redistributed across tiers |
| **What improves over time?** | Fitness ↑, Diversity ↓ (controlled), Strategy allocation optimized | Natural convergence with diversity safeguards |
| **What changes on Tier-2 sharing?** | Individuals migrate between T2 islands | Strategies stay; populations mix |
| **What changes on promotion?** | The algorithm operating on a T2 island | Population stays; only the update mechanism changes |
| **Can an eliminated T1 strategy come back?** | YES — it's never eliminated, always running in T1 | Unlike pure island model, T1 guarantees every strategy gets a chance |

---

## 🎯 FINAL FLOW SUMMARY

```
┌──────────────────────────────────────────────────────────────┐
│                      INITIALIZATION                          │
│  Generate N=50 random individuals from search space          │
│  Evaluate all on dataset → initial fitness                   │
│  Assign: 20 → Tier 1, 30 → Tier 2 (unassigned)             │
│  Split Tier 1: GA(5), PSO(5), DE(5), CMA-ES(5)             │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                   TIER-1 WARM-UP (5 iters)                    │
│  Each strategy evolves its small batch independently         │
│  Compute composite scores for all 4 strategies               │
│  Top 2 → promoted to Tier 2 as islands                      │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│               PARALLEL OPERATION (main loop)                  │
│                                                               │
│  ┌──── Tier 1 ────┐    ┌─────── Tier 2 ───────┐             │
│  │ GA(5) ─────────│    │ Island-1: DE(15)     │             │
│  │ PSO(5) ────────│    │ Island-2: CMA-ES(15) │             │
│  │ DE(5) ─────────│    │                      │             │
│  │ CMA-ES(5) ─────│    │ Every k iters:       │             │
│  │                │    │ Share top 20%        │             │
│  │ All compete    │    │ between islands      │             │
│  │ continuously   │    │                      │             │
│  └───────┬────────┘    └────────┬─────────────┘             │
│          │                      │                            │
│          ▼                      ▼                            │
│  ┌───────────────────────────────────────────┐               │
│  │        CHALLENGE CHECK (every k iters)    │               │
│  │  If T1-strategy score > T2-island score:  │               │
│  │     → Promote T1 strategy to T2           │               │
│  │     → Demoted strategy returns to T1      │               │
│  │     → Grace period for new incumbent      │               │
│  └───────────────────┬───────────────────────┘               │
│                      │                                        │
│  ┌───────────────────▼───────────────────────┐               │
│  │    STAGNATION CHECK (both Tier 1 & 2)     │               │
│  │                                           │               │
│  │  T2 stagnation:                           │               │
│  │    → Inject 20% random individuals        │               │
│  │    → If persists → UCB1 bandit switch     │               │
│  │                                           │               │
│  │  T1 single strategy stagnation:           │               │
│  │    → Refresh bottom 50% of its batch      │               │
│  │    → Temporarily shrink its batch         │               │
│  │                                           │               │
│  │  T1 global stagnation (all strategies):   │               │
│  │    → Keep 1 best per island, refresh rest │               │
│  │    → Re-run warm-up for 3 iters           │               │
│  └───────────────────┬───────────────────────┘               │
│                      │                                        │
└──────────────────────┼───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                      CONVERGENCE                              │
│  Budget exhausted OR target accuracy reached                  │
│  Return: Best individual OR ensemble of top-k                │
└──────────────────────────────────────────────────────────────┘
```

---

> [!TIP]
> **Key Insight:** Tier 1 is your **always-on strategy lab** — all 4 algorithms (GA, DE, PSO, CMA-ES) always have a seat at the table. Tier 2 is where the **real work** happens with the bulk of the population. The promotion mechanism ensures the best algorithm earns the most resources, while sharing ensures knowledge flows between Tier-2 islands. Stagnation is handled at **both tiers** — T2 gets diversity injection + UCB1 bandit, T1 gets batch refresh + temporary rebalancing. This combines the exploration guarantee of Method C, the knowledge sharing of Method B, and the adaptive selection of Method A.
