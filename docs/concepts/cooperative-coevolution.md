# Cooperative Co-evolution (Tiered Island Model)

## The Premature Convergence Problem

In standard evolutionary optimization, populations operating in isolation suffer from **genetic drift**: individuals rapidly homogenize around the first decent local optimum discovered, causing population diversity to collapse to zero.

EvoSynth resolves this using a **Two-Tier Cooperative Co-evolution Island Architecture**:

```text
                  Initial Population (N = 100)
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       Tier 1: Exploration           Tier 2: Reserved
         (40% = 40 inds)              (60% = 60 inds)
     ┌─────┬─────┬─────┬─────┐
     │ GA  │ PSO │ DE  │CMAES│
     └─────┴─────┴─────┴─────┘
               │
               ▼ (Step 4: Composite Scoring + UCB1)
      Top-2 Promoted to Tier 2 Exploitation Islands
               │
      ┌────────┴────────┐
      ▼                 ▼
   Island A          Island B
  (Winner 1)        (Winner 2)
      ▲                 ▲
      └──── Elite ──────┘
          Migration
```

---

## 1. Tier 1: Exploration Tournament

* **Allocation:** 40% of the total population is split equally into 4 distinct islands (GA, PSO, DE, CMA-ES).
* **Initial Execution:** Every algorithm runs for $T_1$ iterations under identical search boundaries.
* **Objective:** Assess which algorithm family aligns best with the geometric topology of the objective dataset.

---

## 2. Tier 2: Exploitation & Reserved Budget

* **Promotion:** After Tier 1 iterations conclude, the Composite Scoring engine ranks the algorithms. The top 2 algorithms are promoted to Tier 2.
* **Injection of Reserved Candidates:** The 60% reserved population individuals are injected into the winning Tier 2 optimizers to expand their swarm capacity for intense local exploitation.

---

## 3. Dynamic Safeguards

### A. Elite Solution Migration
Every $K$ iterations, the top 20% elite individuals from both Tier 2 islands are pooled together, sorted by global fitness, and redistributed across both islands. This forces swarms to recalculate global trajectory vectors and prevents them from stagnating in independent local basins.

### B. Stagnation Detection & Random Latent Injection
If an algorithm's fitness trend remains flat for 3 consecutive checkpoint intervals, the bottom 20% of its population is purged and replaced with freshly generated random latent vectors. This injects fresh entropy and diversity into the search without sacrificing elite performers.

### C. Grace Period & Rollback
If a newly promoted Tier 2 algorithm fails to improve the global best fitness within a grace period, the controller initiates a rollback, re-instating the previous high-performing strategy.
