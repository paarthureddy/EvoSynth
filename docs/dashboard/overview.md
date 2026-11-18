# Dashboard Overview

## The Need for Glass-Box Observability

Traditional AutoML systems operate as opaque black boxes: users kick off a 2-hour job and see only a spinning wheel or raw terminal text logs. If an algorithm gets trapped in a local minimum or if computational resources are wasted on dead strategies, the user has no visibility.

EvoSynth introduces **Glass-Box Telemetry**: a live React dashboard providing real-time visibility into the Multi-Armed Bandit's decision-making process.

---

## Dashboard Components

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        EvoSynth Live Telemetry                         │
├────────────────────────┬───────────────────────────────────────────────┤
│ Tier State & Islands   │ Convergence Metric Curve                      │
│ [T1: DE, PSO]          │ 📈 Best Global Accuracy vs Iterations         │
│ [T2: GA, CMAES]        │                                               │
├────────────────────────┼───────────────────────────────────────────────┤
│ UCB1 Composite Radar   │ Terminal Log Mirror                           │
│ 🎯 Fitness, Velocity,  │ [INFO] Evaluating batch 16/100...             │
│    Diversity, Trend    │ [OK] New Best: 95.33% via PSO                 │
└────────────────────────┴───────────────────────────────────────────────┘
```

1. **Tier State Tracker:** Displays which algorithms occupy Active Tier 1 (Exploration) versus Promoted Tier 2 (Exploitation) islands.
2. **Convergence Curves:** Live chart tracking global accuracy progression and individual algorithm accuracy trajectories over time.
3. **Composite Metrics Table:** Shows the exact numerical breakdown of fitness, speed, diversity, trend, and UCB exploration bonus for every arm.
4. **Live Event Stream:** Instant visual notifications whenever elite migrations, random stagnation injections, or rollbacks occur.
