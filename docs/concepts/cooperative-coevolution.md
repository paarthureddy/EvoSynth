# Cooperative Co-evolution (Tiered Island Model)

## The Premature Convergence Problem

In standard evolutionary optimization, populations operating in isolation suffer from **genetic drift**: individuals rapidly homogenize around the first decent local optimum discovered, causing population diversity to collapse.

EvoSynth addresses this with a **two-tier island model** in which four algorithm families compete, the best two are promoted, and the promoted islands cooperate:

```text
              Initial population (N vectors, evaluated)
                              │
                 stratified split by fitness rank
               ┌──────────────┴──────────────┐
               ▼                             ▼
     Tier 1: 40% of N               Tier 2 reserve: 60% of N
     4 islands of N/10                2 groups of 30% of N
     ┌─────┬─────┬─────┬─────┐
     │ GA  │ PSO │ DE  │CMAES│
     └─────┴─────┴─────┴─────┘
               │  composite score + UCB1
               ▼
   Top 2 families promoted to Tier 2 (each: its Tier 1 island + one reserve group)
      ┌────────┴────────┐
      ▼                 ▼
   Island A          Island B
  (rank 1)          (rank 2)
      ▲                 ▲
      └── pairwise elite sharing ──┘
```

---

## 1. Stratified Tier Split

The evaluated initial population is sorted by fitness and divided into rank strata. Each Tier 1 island and each reserve group receives one member drawn at random from each stratum, so every island starts with a similar mix of good and weak configurations and no optimizer gets a head start. Two alternative splits are available for the ablation: `random` (a plain shuffle) and `elite` (the best 40% to Tier 1 in rank order).

## 2. Tier 1: Competition

The four islands (GA, PSO, DE, CMA-ES) each run $T_1$ generations under identical search bounds. The composite score and the UCB1 bandit then rank the families; the top two are promoted.

## 3. Tier 2: Exploitation

Each promoted family gets a Tier 2 island made of its own Tier 1 island plus one reserve group, so both Tier 2 islands have the same size. The two families that were not promoted keep running in Tier 1 and can be swapped back in.

Every $k$ iterations the controller holds a **checkpoint**: it rescores all islands and applies the safeguards below. With `auto_k`, $k$ is lowered automatically so that Tier 2 reaches at least two checkpoints within the budget.

---

## 4. Safeguards (at each checkpoint)

### A. Pairwise Elite Sharing
The two Tier 2 islands exchange elites directly: island A's worst 20% are replaced by island B's best 20%, and vice versa. Each optimizer updates its own search state (`inject`), for example PSO resets the affected particles and CMA-ES restarts its distribution around the new population.

### B. SWAP
If the best Tier 1 family's UCB score beats the weakest Tier 2 family's by more than the margin (0.10, an absolute difference), the two exchange tiers. The demoted family enters a cooldown of $2k$ iterations before it can be promoted again.

### C. Grace Period and Rollback
A newly swapped-in family has 5 iterations to improve the global best. If it does not, the controller rolls back to the family it replaced.

### D. Random Injection on Stagnation
A Tier 2 island without improvement for 3 consecutive checkpoints has its bottom 20% replaced by fresh random vectors; a Tier 1 island without improvement at a checkpoint has its bottom 50% replaced. The optimizer is told through `inject`, so its internal state stays consistent.

All percentages, limits and margins are settings in `settings/engine.toml` (`[engine]`, `[safeguards]`).
