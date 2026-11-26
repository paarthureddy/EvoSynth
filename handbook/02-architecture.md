# 02 — Architecture

## 1. Repository layout

```
EvoSynth/
├── pyproject.toml               # Package `evosynth` (src layout), pinned dependencies, extras, `evosynth` command
├── requirements.txt             # `-e .[all]`
├── README.md / CONTRIBUTING.md / LICENSE (MIT)
├── settings/                    # ── EVERY TUNABLE VALUE ── (TOML, merged into one set of settings)
│   ├── README.md                # Index: which file holds what, rules, how a value is chosen
│   ├── run.toml                 # [run] seed, data_seed, max_workers
│   ├── engine.toml              # [engine] [bandit] [composite] [safeguards]
│   ├── optimizers.toml          # [optimizers.ga|pso|de|cmaes]
│   ├── search_spaces.toml       # [search_space.tabular|nas|toy]
│   ├── evaluation.toml          # [evaluation] [selection] [features] [refit] [inference]
│   ├── datasets.toml            # [datasets.<name>]
│   ├── telemetry.toml           # [telemetry]
│   └── experiments.toml         # [experiments.search_demo|search_full|scaling|ablation|baselines|nas|speedup]
├── src/evosynth/                # ── THE FRAMEWORK ──
│   ├── __main__.py / cli.py     # `evosynth <command>` (= `python -m evosynth`)
│   ├── config.py                # read_settings, load_config (preset + overrides), validate, get/set_settings
│   ├── paths.py                 # PROJECT_ROOT, DATA_DIR (.data/), RESULTS_DIR (results/), SETTINGS_DIR
│   ├── seeding.py               # run_seed / data_seed, stream(*names): one independent generator per component
│   ├── core/
│   │   ├── individual.py        # Individual(vector, fitness, source_algorithm)
│   │   └── decoding.py          # decode_vector_to_dict(vector, space)
│   ├── search_spaces/           # tabular.py (8-D), nas.py (7-D), toy.py (4-D), built from settings
│   ├── datasets/                # registry.py (spec, load, partition), loaders.py (sklearn / OpenML / torchvision → .data/), splits.py (test first)
│   ├── features/generic.py      # GenericFeatureEngineer: none / stats / interactions
│   ├── models/
│   │   ├── factory.py           # build_model (search / refit), build_default_model, fit_model (early stopping, relabelling)
│   │   ├── refit.py             # refit_split, refit_and_score: the one refit for every tuned method
│   │   ├── metrics.py           # score(model, X, y, metric): accuracy / balanced_accuracy / roc_auc
│   │   ├── inference.py         # measure(model, X): latency per row, rows/s, model size
│   │   └── cnn.py               # DynamicCNN
│   ├── evaluators/
│   │   ├── tabular.py           # TabularEvaluator(dataset, subsample_size)
│   │   ├── image.py             # ImageCNNEvaluator (PyTorch NAS)
│   │   └── toy.py               # DummyObjectiveFunction (fast test function)
│   ├── engine/
│   │   ├── controller.py        # DynamicOptimizer: the whole meta-optimization loop
│   │   ├── scoring.py           # CompositeScoringEngine (5 weighted, min-max-normalised metrics)
│   │   ├── bandit.py            # UCB1Bandit: scores and ranks the families (promotion, SWAP)
│   │   └── tiers.py             # TierManager: stratified / random / elite split, pairwise elite sharing
│   ├── optimizers/              # base.py (ask / tell / inject, own rng), ga.py, pso.py, de.py, cmaes.py
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
│       ├── selection.py         # confirm_best: confirm before select
│       ├── summary.py           # `evosynth summary`: tables over seeds, convergence curves
│       ├── speedup.py           # `evosynth speedup`
│       └── nas.py               # `evosynth nas`
├── tools/run_experiments.py     # Experiment plans (smoke / lean / core), resumable
├── tools/export_splits.py       # Exports the exact dataset splits (EvoSynth-Bench release)
├── results/                     # One folder per run (see §4); archive/ holds the v1 panel-review results
├── dashboard/                   # React 19 + Vite 8 + Tailwind 4 live dashboard (src/App.jsx)
├── docs/                        # VitePress documentation site
│   ├── guide/ concepts/ algorithms/ reference/ dashboard/ developer/  (site pages)
│   ├── research/                # comparative_analysis.md, AutoML_Relevant_Papers.txt
│   └── public/                  # logo.png, favicon.png, arch.png
├── paper/                       # paper.tex, slides.tex, figures/, links.md
├── handbook/                    # This handbook + panel-defense guide (internal; not part of the docs site)
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
  │   evaluator = TabularEvaluator(dataset, n)     # loads the dataset's search rows once (test rows held out)
  │   opt = DynamicOptimizer(space, evaluator, population_csv=...)   # budget etc. from [engine]
  ▼
opt.run()
  STEP 0  print config; emit step 0
  STEP 1  for chunks of size 2·max_workers: random vectors → _evaluate_batch('Init')
  STEP 2  TierManager(split="stratified"): 40% → 4 Tier 1 islands, 60% → 2 reserve groups; k sized (auto_k)
  STEP 3  for i in 1..t1_iters:
             for each algo: inds = _evaluate_batch(opt.ask(island), 'T1_algo'); opt.tell(inds)
             every k: composite → island without improvement: inject 50% random (optimizer.inject)
  STEP 4  composite + UCB1Bandit over T1 → rank → top2 → build T2 optimizers (T1 island + one reserve group each)
  STEP 5  while evals_used < budget:
             each T2 and T1 optimizer: ask(pop_size) → evaluate → tell
             grace-period bookkeeping (rollback if failed)
             every k: score all (scoring_history) → maybe SWAP → stagnation injection → pairwise elite sharing
  STEP 6  print totals, winning algo tag, decoded best hyperparameters; return best Individual
  ▼
experiment (search / scaling)
  confirm_best: top 5 distinct configs re-scored with 5-fold CV → selected config
  refit_and_score: GenericFeatureEngineer (fit on train) → build model → fit on train rows
                   (boosted: early stop on a train slice) → score test rows → measure inference → model.pkl
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

`results/<experiment>/<dataset>/<variant>/seed_<run.seed>/`

| File | Contents |
|---|---|
| `manifest.json` | Command line, every resolved setting, git commit/branch/dirty flag, Python and library versions, start/end time, status (`running` / `completed` / `failed` / `interrupted`) |
| `result.json` | The run's final numbers (metric, best and selected CV score, confirmation table, selected configuration, held-out test score, inference cost, failed folds, …) |
| `run.log` | Everything printed to the console (UTF-8). `sys.stdout` is replaced by `LoggerWriter` for the duration of the run |
| `events.jsonl` | Every engine event (`step`, `progress`, `tier_state`, `metric`, `composite`, `switch`) with a timestamp — the same stream the dashboard receives, saved even when the server is not running |
| `population.csv` | One row per evaluation: `Eval_ID,Phase,Algorithm,Fitness,<hyperparameters...>` (Phase = Init / T1 / T2) |
| `model.pkl` | The refitted model, its feature engineer and configuration (search, scaling, Random Search, Optuna) |
| `checkpoint.json`, `checkpoints/` | Resume state (scaling, ablation; the ablation saves each optimizer's full state) |

Only `manifest.json`, `result.json` and `summary/` folders are committed to git. Experiment-specific outputs are listed in [07-experiments.md](07-experiments.md).

**Telemetry.** On first use (`get_reporter()`, or when a run starts) an `EventReporter` tries `GET http://localhost:8000/docs` with a 0.5 s timeout; live telemetry is enabled if that succeeds. If the server is not up yet, the reporter re-probes every `telemetry.reprobe_seconds` (10 s) in the background, so the dashboard can also be started after a run has begun.

## 5. Concurrency model

- **Evaluation parallelism.** Each batch is evaluated with threads. XGBoost, LightGBM and sklearn release the GIL inside native training code, so threads give real parallelism. Each model uses `n_jobs = evaluation.model_n_jobs` (1) to avoid oversubscription. Final refits use `refit.model_n_jobs` (-1).
- **Telemetry.** Each `emit` appends to `events.jsonl` under a lock and starts a daemon thread that POSTs with a 1 s timeout and ignores failures, so telemetry never blocks or crashes the optimizer.
- **Optimizers.** They are single-threaded. `ask` and `tell` run on the main thread, between parallel evaluation batches.

## 6. Configuration surface

`DynamicOptimizer(search_space, evaluator, budget=None, pop_size=None, t1_iters=None, k_iters=None, swap_margin=None, stagnation_limit=None, elite_pct=None, inject_pct=None, max_workers=None, population_csv=None)`

Every argument left as `None` is taken from the settings:

| Parameter | Setting | Default | Meaning |
|---|---|---|---|
| `search_space` | — | — | ConfigSpace object; its keys define dimension order |
| `evaluator` | — | — | Object with `.evaluate(ind)` |
| `budget` | `engine.budget` | 2000 | Total evaluations |
| `pop_size` | `engine.pop_size` | 100 | Initial population N (≥ 40 so each Tier 1 island has ≥ `engine.min_island_size` = 4) |
| `t1_iters` | `engine.t1_iters` | 30 | Tier 1 tournament iterations |
| `k_iters` | `engine.k_iters` | 10 | Checkpoint interval k (upper bound when `engine.auto_k`) |
| `swap_margin` | `safeguards.swap_margin` | 0.10 | Additive UCB margin needed for a SWAP |
| `stagnation_limit` | `safeguards.stagnation_limit` | 3 | Consecutive checkpoints without improvement before T2 injection |
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
| Trend window | `composite.trend_window` | 3 scoring calls |
| Tier split method | `engine.tier_split` | `stratified` (also `random`, `elite`) |
| Automatic k | `engine.auto_k`, `engine.min_t2_checkpoints` | on, ≥ 2 Tier 2 checkpoints |

The commands apply presets on top (e.g. `evosynth run` uses `[experiments.search_demo]`: budget 200, pop 40, t1 3, k 2; `--full` uses budget 500). Settings are validated when loaded (`config.validate`).

---

## Quiz

**1.** Which folder holds every tunable value, and how are its files combined?
<details><summary>Answer</summary>`settings/`; every `*.toml` file is merged into one set of settings, and a top-level section may be defined in only one file.</details>

**2.** In what order is a setting's value resolved?
<details><summary>Answer</summary>settings/*.toml → the command's preset → command-line flags → `--set` overrides (later wins).</details>

**3.** Name the module that creates one independent random generator per component, and why it matters.
<details><summary>Answer</summary>`src/evosynth/seeding.py` (`stream(*names)`); it makes a run reproducible from `run.seed` regardless of thread timing, and draws in one component never shift another's numbers.</details>

**4.** List the files a search run writes to its run folder.
<details><summary>Answer</summary>`manifest.json`, `result.json`, `run.log`, `events.jsonl`, `population.csv` and `model.pkl` (plus checkpoints for scaling and ablation).</details>

**5.** Which two files from results are committed to git?
<details><summary>Answer</summary>`manifest.json` and `result.json` (plus `summary/` folders); logs, CSVs and models are not.</details>

**6.** What is the single path through which every evaluation passes, and name two things it guarantees.
<details><summary>Answer</summary>`_evaluate_batch`: the budget is never exceeded (requests are truncated), and every evaluation is counted, logged to population.csv and compared with the global best.</details>

**7.** Why are evaluations run in threads rather than processes, and is the speed-up real?
<details><summary>Answer</summary>XGBoost, LightGBM and scikit-learn release the GIL in native training code, so threads run truly in parallel while sharing the loaded data.</details>

**8.** What does `engine.auto_k` do?
<details><summary>Answer</summary>Lowers the checkpoint interval k so that Tier 2 reaches at least `min_t2_checkpoints` (2) checkpoints within the budget.</details>

**9.** Can the dashboard be started after a run has begun? Why?
<details><summary>Answer</summary>Yes; the EventReporter re-probes the server every 10 s while it is unreachable, and all events are also saved to `events.jsonl`.</details>

**10.** Which module holds the one refit used by every tuned method?
<details><summary>Answer</summary>`src/evosynth/models/refit.py` (`refit_and_score`).</details>
