# 02 — Architecture

## 1. Repository layout

```
EvoSynth/
├── pyproject.toml               # Package `evosynth` (src layout), pinned dependencies, extras, `evosynth` command
├── requirements.txt             # `-e .[all]`
├── README.md / CONTRIBUTING.md / LICENSE (MIT)
├── settings/                    # ── EVERY TUNABLE VALUE ── (TOML, merged into one set of settings)
│   ├── README.md                # Index: which file holds what, rules, how a value is chosen
│   ├── run.toml                 # [run] random_state, max_workers
│   ├── engine.toml              # [engine] [bandit] [composite] [safeguards]
│   ├── optimizers.toml          # [optimizers.ga|pso|de|cmaes]
│   ├── search_spaces.toml       # [search_space.tabular|nas|toy]
│   ├── evaluation.toml          # [evaluation] [features] [refit]
│   ├── datasets.toml            # [datasets.<name>]
│   ├── telemetry.toml           # [telemetry]
│   └── experiments.toml         # [experiments.search_demo|search_full|scaling|ablation|baselines|nas]
├── src/evosynth/                # ── THE FRAMEWORK ──
│   ├── __main__.py / cli.py     # `evosynth <command>` (= `python -m evosynth`)
│   ├── config.py                # read_settings, load_config (preset + overrides), validate, get/set_settings
│   ├── paths.py                 # PROJECT_ROOT, DATA_DIR (.data/), RESULTS_DIR (results/), SETTINGS_DIR
│   ├── core/
│   │   ├── individual.py        # Individual(vector, fitness, source_algorithm)
│   │   └── decoding.py          # decode_vector_to_dict(vector, space)
│   ├── search_spaces/           # tabular.py (9-D), nas.py (7-D), toy.py (4-D), built from settings
│   ├── datasets/                # registry.py (name → loader + spec), loaders.py (download to .data/), splits.py
│   ├── features/generic.py      # GenericFeatureEngineer: none / stats / interactions
│   ├── models/
│   │   ├── factory.py           # build_model / fit_model for search, refit and baseline phases
│   │   ├── refit.py             # refit_split, refit_and_score (60k train / 20k test)
│   │   └── cnn.py               # DynamicCNN
│   ├── evaluators/
│   │   ├── tabular.py           # TabularEvaluator(dataset, subsample_size)
│   │   ├── image.py             # ImageCNNEvaluator (PyTorch NAS)
│   │   └── toy.py               # DummyObjectiveFunction (fast test function)
│   ├── engine/
│   │   ├── controller.py        # DynamicOptimizer: the whole meta-optimization loop
│   │   ├── scoring.py           # CompositeScoringEngine (5 weighted, min-max-normalised metrics)
│   │   ├── bandit.py            # UCB1Bandit class (bookkeeping; see 03 §8)
│   │   └── tiers.py             # TierManager helper (not yet used by the controller)
│   ├── optimizers/              # base.py (ask/tell contract), ga.py, pso.py, de.py, cmaes.py
│   ├── telemetry/
│   │   ├── reporter.py          # EventReporter: events.jsonl + optional POST to the server
│   │   ├── run_logger.py        # stdout mirror (run.log), PopulationLogger (population.csv)
│   │   └── server.py            # FastAPI SSE relay (/event, /stream)
│   └── experiments/
│       ├── run_context.py       # RunContext: one folder per run, manifest.json / result.json
│       ├── checkpointing.py     # atomic JSON / pickle writes
│       ├── search.py            # `evosynth run`
│       ├── scaling.py           # `evosynth scale`
│       ├── ablation.py          # `evosynth ablation`
│       ├── baselines.py         # `evosynth baselines`
│       └── nas.py               # `evosynth nas`
├── tools/run_experiments.ps1    # Staged PowerShell orchestrator for the Covertype study
├── results/                     # One folder per run (see §4); archive/ holds the v1 panel-review results
├── dashboard/                   # React 19 + Vite 8 + Tailwind 4 live dashboard (src/App.jsx)
├── docs/                        # VitePress documentation site
│   ├── guide/ concepts/ algorithms/ reference/ dashboard/ developer/  (site pages)
│   ├── handbook/                # This handbook (excluded from the site)
│   ├── research/                # comparative_analysis.md, PANEL_DEFENSE_MASTER_GUIDE.md, AutoML_Relevant_Papers.txt
│   └── public/                  # logo.png, favicon.png, arch.png
├── paper/                       # paper.tex, slides.tex, figures/, links.md
├── .github/workflows/deploy-docs.yml  # Builds and deploys docs to GitHub Pages
└── .data/                       # Downloaded datasets (git-ignored)
```

**Installation.** `pip install -e ".[all]"` installs the package in editable mode and the `evosynth` command. There are no root scripts: every experiment is a sub-command (see [12-setup-and-usage.md](12-setup-and-usage.md)).

## 2. Module dependency graph

```
cli.py ──► config (load settings) ──► experiments/<command>.py
                                          ├──► run_context.RunContext ──► telemetry.reporter / run_logger
                                          ├──► search_spaces.* ──► ConfigSpace
                                          ├──► evaluators.* ──► datasets / features / models ──► sklearn / xgboost / lightgbm / torch
                                          └──► engine.controller.DynamicOptimizer
                                                     ├──► engine.scoring.CompositeScoringEngine
                                                     ├──► engine.bandit.UCB1Bandit
                                                     ├──► optimizers.{ga,pso,de,cmaes} ──► optimizers.base.BaseOptimizer
                                                     │                                   └► cma (CMA-ES only)
                                                     ├──► core.individual / core.decoding
                                                     └──► telemetry.reporter.get_reporter() ──HTTP POST──► telemetry.server ──SSE──► dashboard
```

Every module reads its constants from the active settings (`evosynth.config.get_settings()`); no Python file holds its own copy of a tunable value.

The **evaluator contract** is duck-typed: any object with a `.space` (a ConfigSpace) and `.evaluate(individual) -> float` (higher is better, conventionally 0–100) works.

The **optimizer contract** is also small:
- `ask(n) -> list[np.ndarray]`
- `tell(list[Individual])`
- `.population` (a list of Individuals, best first)
- `.pop_size`

## 3. Runtime data flow, one run

```
evosynth run  (cli.cmd_run)
  │ settings = load_config(preset, flags, --set)   # settings/*.toml → preset → flags → --set
  │ with RunContext("search", dataset, variant):   # creates the run folder, manifest.json, run.log
  │   space = get_tabular_search_space(...)        # ConfigSpace, d dims
  │   evaluator = TabularEvaluator(dataset, n)     # loads/subsamples data once
  │   opt = DynamicOptimizer(space, evaluator, population_csv=...)   # budget etc. from [engine]
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
experiment (search / scaling)
  refit_and_score: decode best → GenericFeatureEngineer → build final model → fit 60k → test 20k
  run.write_result({...})                          # result.json
```

### `_evaluate_batch(vectors, algo_name)`, the single path to an evaluation

1. Truncate `vectors` to the remaining budget (`budget − evals_used`). The budget can never be exceeded.
2. Wrap each vector in an `Individual`, call `evaluator.evaluate`, and set its fitness and `source_algorithm`. This runs in a `ThreadPoolExecutor` with `min(max_workers, len(vectors))` workers, where `max_workers` comes from `run.max_workers` (0 = `min(8, os.cpu_count())`).
3. For each result, in order:
   - increment `evals_used` and that algorithm's `algo_evals`;
   - append a row to `population.csv` (when a path was given);
   - update the per-algorithm best and emit a `metric` event when it improves;
   - update the global `best_individual`;
   - append to `history` (`eval_id`, `fitness`, `best_so_far`, `algo`).

## 4. Files written by a run

Importing any `evosynth` module has **no side effects**. Everything is set up by `RunContext` when a command starts a run, in one folder per run:

`results/<experiment>/<dataset>/<variant>/seed_<random_state>/`

| File | Contents |
|---|---|
| `manifest.json` | Command line, every resolved setting, git commit/branch/dirty flag, Python and library versions, start/end time, status (`running` / `completed` / `failed` / `interrupted`) |
| `result.json` | The run's final numbers (best CV score, best configuration, refit test accuracy, …) |
| `run.log` | Everything printed to the console (UTF-8). `sys.stdout` is replaced by `LoggerWriter` for the duration of the run |
| `events.jsonl` | Every engine event (`step`, `progress`, `tier_state`, `metric`, `composite`, `switch`) with a timestamp — the same stream the dashboard receives, saved even when the server is not running |
| `population.csv` | One row per evaluation: `Eval_ID,Phase,Algorithm,Fitness,<hyperparameters...>` (Phase = Init / T1 / T2) |
| `checkpoint.json`, `checkpoints/` | Resume state (scaling, ablation) |

Only `manifest.json`, `result.json` and `summary/` folders are committed to git. Experiment-specific outputs are listed in [07-experiments.md](07-experiments.md).

**Telemetry.** On first use (`get_reporter()`, or when a run starts) an `EventReporter` tries `GET http://localhost:8000/docs` with a 0.5 s timeout; live telemetry is enabled only if that succeeds. **Start `evosynth serve` before starting a run** to see it on the dashboard.

## 5. Concurrency model

- **Evaluation parallelism.** Each batch is evaluated with threads. XGBoost, LightGBM and sklearn release the GIL inside native training code, so threads give real parallelism. Each model uses `n_jobs = evaluation.model_n_jobs` (1) to avoid oversubscription. Final refits use `refit.model_n_jobs` (-1).
- **Telemetry.** Each `emit` appends to `events.jsonl` under a lock and starts a daemon thread that POSTs with a 1 s timeout and ignores failures, so telemetry never blocks or crashes the optimizer.
- **Optimizers.** They are single-threaded. `ask` and `tell` run on the main thread, between parallel evaluation batches.

## 6. Configuration surface

`DynamicOptimizer(search_space, evaluator, budget=None, pop_size=None, t1_iters=None, k_iters=None, swap_threshold=None, stagnation_limit=None, elite_pct=None, inject_pct=None, max_workers=None, population_csv=None)`

Every argument left as `None` is taken from the settings:

| Parameter | Setting | Default | Meaning |
|---|---|---|---|
| `search_space` | — | — | ConfigSpace object; its keys define dimension order |
| `evaluator` | — | — | Object with `.evaluate(ind)` |
| `budget` | `engine.budget` | 2000 | Total evaluations |
| `pop_size` | `engine.pop_size` | 100 | Initial population N |
| `t1_iters` | `engine.t1_iters` | 30 | Tier 1 tournament iterations |
| `k_iters` | `engine.k_iters` | 10 | Checkpoint interval k |
| `swap_threshold` | `safeguards.swap_margin` | 0.10 | Additive UCB margin needed for a SWAP |
| `stagnation_limit` | `safeguards.stagnation_limit` | 3 | Consecutive zero-trend checkpoints before T2 injection |
| `elite_pct` | `safeguards.elite_pct` | 0.20 | Fraction shared between T2 islands |
| `inject_pct` | `safeguards.t2_inject_pct` | 0.20 | Fraction replaced on T2 stagnation |
| `max_workers` | `run.max_workers` | 0 → `min(8, cpu_count)` | Evaluation threads |
| `population_csv` | — | `None` | Where to write one row per evaluation |

Further constants, all in `settings/engine.toml`:

| Constant | Setting | Value |
|---|---|---|
| Exploration coefficient `c` | `bandit.c` | 0.1 |
| Tier 1 share | `engine.t1_fraction` | 0.4 |
| Number of top optimizers promoted | `engine.n_promoted` | 2 |
| T1 injection fraction | `safeguards.t1_inject_pct` | 0.50 |
| Grace period | `safeguards.grace_iters` | 5 iterations |
| Cooldown | `safeguards.cooldown_multiplier` | 2 → `2·k` |
| Composite weights | `composite.*` | 0.30 / 0.25 / 0.15 / 0.15 / 0.15 |

The commands apply presets on top (e.g. `evosynth run` uses `[experiments.search_demo]`: budget 120, pop 20, t1 3, k 2). Settings are validated when loaded (`config.validate`).
