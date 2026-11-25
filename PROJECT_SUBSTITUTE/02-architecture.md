# 02 — Architecture

## 1. Repository layout

```
EvoSynth/
├── main.py                      # Root wrapper → scripts/main.py (pathological noisy demo)
├── run_covertype.py             # Root wrapper → scripts/run_covertype.py (Covertype demo + refit)
├── run_vision_nas.py            # Root wrapper → scripts/run_vision_nas.py (CNN NAS)
├── api_server.py                # Root wrapper → scripts/api_server.py (starts uvicorn on :8000)
├── run_experiments.ps1          # Staged PowerShell orchestrator for all Covertype experiments
├── requirements.txt             # Python dependencies
├── README.md / CONTRIBUTING.md / LICENSE (MIT)
├── Hybrid_Workflow.md           # Long-form simulated walkthrough of a run (also in docs/)
├── "Hybrid Workflow.pdf"        # PDF rendering of the above (made by scripts/generate_hybrid_workflow_pdf.py)
├── assets/
│   ├── Logo.png                 # Project logo
│   └── arch.png                 # End-to-end architecture figure
├── automl_engine/               # ── THE FRAMEWORK ──
│   ├── core/
│   │   ├── individual.py        # Individual(vector, fitness, source_algorithm)
│   │   ├── search_space.py      # ConfigSpace definitions (dummy, tabular 9-D, NAS 7-D)
│   │   ├── evaluator.py         # DummyObjectiveFunction, XGBoostEvaluator (breast cancer)
│   │   ├── hard_evaluator.py    # PathologicalXGBoostEvaluator (30% label noise)
│   │   ├── covtype_evaluator.py # CovertypeEvaluator (UCI Forest Covertype)
│   │   ├── image_evaluator.py   # DynamicCNN + ImageCNNEvaluator (PyTorch NAS)
│   │   └── generic_feature_engineer.py # none / stats / interactions
│   ├── engine/
│   │   ├── controller.py        # DynamicOptimizer: whole meta-optimization loop + telemetry + logging
│   │   ├── composite.py         # CompositeScoringEngine (5 weighted, min-max-normalised metrics)
│   │   ├── ucb1.py              # UCB1Bandit class (bookkeeping; see 03 §8)
│   │   ├── tier_manager.py      # TierManager helper (not used by controller)
│   │   └── population.py        # PopulationManager helper (not used by controller)
│   └── optimizers/
│       ├── base.py              # BaseOptimizer ask/tell contract
│       ├── ga.py                # GeneticAlgorithm
│       ├── pso.py               # ParticleSwarm
│       ├── de.py                # DifferentialEvolution
│       └── cmaes.py             # CMAESOptimizer (wraps the `cma` library)
├── scripts/
│   ├── main.py                  # Pathological demo (fast / --full)
│   ├── api_server.py            # FastAPI SSE telemetry relay
│   ├── run_covertype.py         # Covertype demo/full + 60k/20k refit
│   ├── run_covertype_evosynth.py# Covertype at 120/300/500/1000 budgets, checkpointing, refit
│   ├── run_baselines_forest.py  # Default ExtraTrees, Random Search, Optuna TPE, FLAML
│   ├── run_ablation_study.py    # UPGS-120 / UPGS-2500 ablation (isolated vs EvoSynth)
│   ├── run_ablation_study_forest.py # CVGS-300 / CVGS-500 ablation on Covertype (resumable)
│   ├── run_vision_nas.py        # CNN NAS demo
│   ├── load_dataset.py          # Loader/validator for the released EvoSynth-Bench CSVs
│   ├── generate_hybrid_workflow_pdf.py # ReportLab PDF generator for the workflow spec
│   └── Test1_19_Nov/            # Snapshot copy of the scripts used for the 19-Nov run
├── dashboard/                   # React 19 + Vite 8 + Tailwind 4 live dashboard (src/App.jsx)
├── docs/                        # VitePress documentation site + long-form docs
│   ├── guide/ concepts/ algorithms/ reference/ dashboard/ developer/  (site pages)
│   ├── comparative_analysis.md, PANEL_DEFENSE_MASTER_GUIDE.md, Hybrid_Workflow.md
│   └── AutoML_Relevant_Papers.txt
├── .github/workflows/deploy-docs.yml  # Builds and deploys docs to GitHub Pages
├── .data/covertype/             # Cached sklearn Covertype download (samples_py3, targets_py3)
├── results/                     # UPGS ablation outputs + early Covertype log
│   ├── ablation_upgs_{120,2500}_{run,summary}.txt
│   ├── individual_runs/UPGS-{120,2500}_{GA,PSO,DE,CMAES,EvoSynth}.txt
│   ├── covertype_run_with_fe.txt
│   └── (generated at runtime) run_output.txt, csv/*.csv
├── results_forest/Test1_19_Nov/ # Covertype experiment outputs (19 Nov run)
│   ├── baselines/  evosynth/  ablation/cvgs_300  ablation/cvgs_500  summary/
└── submission/                  # Paper (G002_Paper.tex), slide deck (maint.tex), review PDF, link files
```

**Root wrappers.** Each root `*.py` file imports `main` (or `app`) from `scripts/` so it can run from the repo root. Each script in `scripts/` walks up the directory tree until it finds `automl_engine/`, then puts that directory on `sys.path`. Scripts therefore work from any working directory.

## 2. Module dependency graph

```
scripts/*.py ──► automl_engine.core.search_space ──► ConfigSpace
            ├──► automl_engine.core.<evaluator> ──► sklearn / xgboost / lightgbm / torch
            │                                    └► core.generic_feature_engineer (Covertype only)
            └──► automl_engine.engine.controller.DynamicOptimizer
                        ├──► engine.composite.CompositeScoringEngine
                        ├──► engine.ucb1.UCB1Bandit
                        ├──► engine.tier_manager.TierManager   (imported, unused)
                        ├──► optimizers.{ga,pso,de,cmaes}  ──► optimizers.base.BaseOptimizer
                        │                                   └► cma (CMA-ES only)
                        ├──► core.individual.Individual
                        └──► requests ──HTTP POST──► scripts/api_server.py ──SSE──► dashboard
```

The **evaluator contract** is duck-typed: any object with a `.space` (a ConfigSpace) and `.evaluate(individual) -> float` (higher is better, conventionally 0–100) works.

The **optimizer contract** is also small:
- `ask(n) -> list[np.ndarray]`
- `tell(list[Individual])`
- `.population` (a list of Individuals, best first)
- `.pop_size`

## 3. Runtime data flow, one run

```
entry script
  │ space = get_xgboost_search_space(...)          # ConfigSpace, d dims
  │ evaluator = CovertypeEvaluator(...)            # loads/subsamples data once
  │ opt = DynamicOptimizer(space, evaluator, budget, pop_size, t1_iters, k_iters, ...)
  ▼
opt.run()
  STEP 0  print config; emit step 0
  STEP 1  for chunks of size 2·max_workers: random vectors → _evaluate_batch('Init')
  STEP 2  sort pop desc; T1 = top 40% split in 4 contiguous slices; T2 reserve = rest
  STEP 3  for i in 1..t1_iters:
             for each algo: inds = _evaluate_batch(opt.ask(island), 'T1_algo'); opt.tell(inds)
             every k: composite → if trend==0: inject 50% random into that T1 island
  STEP 4  composite+UCB over T1 → top2 → build T2 optimizers (T1 island + half reserve each)
  STEP 5  while evals_used < budget:
             each T2 and T1 optimizer: ask(pop_size) → evaluate → tell
             grace-period bookkeeping (rollback if failed)
             every k: score all → maybe SWAP → stagnation injection (T2) → elite sharing (T2)
  STEP 6  print totals, winning algo tag, decoded best hyperparameters; return best Individual
  ▼
entry script (Covertype scripts only)
  decode best → GenericFeatureEngineer → build final model → fit 60k → test 20k → print summary
```

### `_evaluate_batch(vectors, algo_name)`, the single path to an evaluation

1. Truncate `vectors` to the remaining budget (`budget − evals_used`). The budget can never be exceeded.
2. Wrap each vector in an `Individual`, call `evaluator.evaluate`, and set its fitness and `source_algorithm`. This runs in a `ThreadPoolExecutor` with `min(max_workers, len(vectors))` workers, where `max_workers` defaults to `min(8, os.cpu_count())`.
3. For each result, in order:
   - increment `evals_used` and that algorithm's `algo_evals`;
   - append a CSV row;
   - update the per-algorithm best and emit a `metric` event when it improves;
   - update the global `best_individual`;
   - append to `history` (`eval_id`, `fitness`, `best_so_far`, `algo`).

## 4. Side effects at import and at run time

Importing `automl_engine.engine.controller` has side effects:

- `results/` and `results/csv/` are created under the repo root.
- `EventReporter()` is instantiated. It tries `GET http://localhost:8000/docs` with a 0.5 s timeout, and telemetry is enabled only if that succeeds. **The API server must therefore be running before the engine is imported.**
- `sys.stdout` is replaced by `LoggerWriter`. From then on every `print` is:
  - echoed to the console,
  - written and flushed to `results/run_output.txt`,
  - sent as an SSE `log` event (non-empty lines only).

When `DynamicOptimizer` is constructed, `PopulationLogger` opens three CSVs:
- `results/csv/warmup_population.csv`
- `results/csv/tier1_population.csv`
- `results/csv/tier2_population.csv`

Each has the header `Eval_ID,Algorithm,Fitness,<hyperparameter names...>` and one row per evaluation, with decoded values.

Experiment scripts write further outputs. See [07-experiments.md](07-experiments.md) for the full list.

## 5. Concurrency model

- **Evaluation parallelism.** Each batch is evaluated with threads. XGBoost, LightGBM and sklearn release the GIL inside native training code, so threads give real parallelism. Each model uses `n_jobs=1` to avoid oversubscription. Final refits use `n_jobs=-1`.
- **Telemetry.** Each `emit` starts a daemon thread that POSTs with a 1 s timeout and ignores failures, so telemetry never blocks or crashes the optimizer.
- **Optimizers.** They are single-threaded. `ask` and `tell` run on the main thread, between parallel evaluation batches.

## 6. Configuration surface (`DynamicOptimizer.__init__`)

| Parameter | Default | Meaning |
|---|---|---|
| `search_space` | — | ConfigSpace object; its keys define dimension order |
| `evaluator` | — | Object with `.evaluate(ind)` |
| `budget` | 2000 | Total evaluations |
| `pop_size` | 100 | Initial population N |
| `t1_iters` | 30 | Tier 1 tournament iterations |
| `k_iters` | 10 | Checkpoint interval k |
| `swap_threshold` | 0.10 | Additive UCB margin needed for a SWAP |
| `stagnation_limit` | 3 | Consecutive zero-trend checkpoints before T2 injection |
| `elite_pct` | 0.20 | Fraction shared between T2 islands |
| `inject_pct` | 0.20 | Fraction replaced on T2 stagnation |
| `max_workers` | `min(8, cpu_count)` | Evaluation threads |

Fixed internal constants:

| Constant | Value |
|---|---|
| Exploration coefficient `c` | 0.1 |
| Tier 1 share | 0.4 |
| T1 injection fraction | 0.50 |
| Grace period | 5 iterations |
| Cooldown | `2·k` |
| Number of top optimizers promoted | 2 |
