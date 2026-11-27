# 12 — Setup & Usage

## 1. Prerequisites

| Component | Version |
|---|---|
| Python | 3.11+ (`requires-python` in `pyproject.toml`). Developed and run on Python 3.14 (Windows). |
| Node.js | 18+ for the dashboard; 20 for the docs CI |
| OS | Developed on Windows. The Python code (including `tools/run_experiments.py`) is OS-agnostic. |
| Hardware | CPU only. Reference machine: Intel i5-12450HX (8 cores / 12 threads), 16 GB RAM. |

## 2. Installation

```bash
git clone https://github.com/paarthureddy/EvoSynth.git
cd EvoSynth
python -m venv venv
venv\Scripts\activate            # Windows   (Linux/macOS: source venv/bin/activate)
pip install -e ".[all]"          # the package + every extra; `pip install -e .` for the core only
```

This installs the `evosynth` command. If its folder is not on `PATH`, use `python -m evosynth` (on Windows with several Pythons: `py -3.14 -m evosynth`).

Dependencies are pinned in `pyproject.toml`:

| Package | Purpose |
|---|---|
| `ConfigSpace` | Search-space definitions and vector ↔ configuration decoding |
| `numpy`, `scipy`, `pandas` | Numerics and dataset loading |
| `scikit-learn` | Covertype download, CV, RandomForest and ExtraTrees |
| `xgboost`, `lightgbm` | Gradient-boosted models |
| `cma` | CMA-ES implementation |
| `fastapi`, `uvicorn`, `requests` | SSE telemetry server and engine → server event posts |
| extra `nas`: `torch`, `torchvision` | CNN NAS |
| extra `baselines`: `optuna`, `flaml` | Competitor baselines |
| extra `dev`: `pytest` | Tests |

**Check the install:**

```bash
evosynth settings check        # validates settings/ (13 sections)
evosynth datasets list         # registered datasets
```

**Datasets.** Each dataset in `settings/datasets.toml` is downloaded on first use into `<repo>/.data/` (git-ignored), or all at once with `evosynth datasets download` (about 230 MB for all eight). Covertype comes from `sklearn.datasets.fetch_covtype`, Fashion-MNIST from torchvision, the others from OpenML.

## 3. Running EvoSynth

Run all commands from the repo root. Every command accepts:
- `--set SECTION.KEY=VALUE` (repeatable) to override any setting for this run, e.g. `--set bandit.c=0.2`;
- `--settings PATH` to use another settings folder.

Experiment commands also accept `--results-dir PATH` (write under PATH instead of `results/`); `run`, `scale` and `nas` accept `--variant NAME` (name of the run folder). Each run writes to `results/<experiment>/<dataset>/<variant>/seed_<random_state>/` (see §4).

### 3.1 A search with refit: the headline demo

```bash
evosynth run                                  # preset search_demo: 200 evals, pop 40, t1=3, k=2, Covertype
evosynth run --full --dataset adult           # preset search_full: 500 evals, pop 40, t1=5, k=3
evosynth run --budget 300 --t1-iters 4 --subsample 10000 --seed 7
evosynth run --no-refit                       # search and selection only
```

**Output** (`results/search/<dataset>/<preset>/seed_<k>/`): Steps 0–6 are printed, then confirm-before-select and a summary box: search CV score (selected and confirmed), held-out test score, architecture, FE strategy and the measured inference cost. `model.pkl` holds the refitted model.

### 3.2 Budget scaling, with checkpointing

```bash
evosynth scale                 # budget 500 (default)
evosynth scale --budget 1000 --dataset higgs --checkpoint-interval 10
```

- `--budget` must be one of `experiments.scaling.budgets` (300, 500, 1000). `t1_iters` and `k_iters` follow the schedule in `[experiments.scaling]`.
- **Outputs** (`results/scaling/<dataset>/budget_<N>/seed_<k>/`): `run.log`; `checkpoint.json`, updated every 10 evaluations and finalised with the test score; `result.json`; `model.pkl`.

### 3.3 Baselines (same splits, same evaluator and refit as EvoSynth)

```bash
evosynth baselines --test all --dataset covertype
evosynth baselines --test random --trials 500        # match the EvoSynth budget
evosynth baselines --test optuna --trials 500        # resumes from its SQLite study
evosynth baselines --test flaml --time-budget 900    # match the EvoSynth wall time
evosynth baselines --test defaults                   # untuned XGBoost / LightGBM / RF / ET
```

- `--test` accepts `all`, `defaults`, `random`, `optuna` or `flaml`.
- **Outputs:** one folder per baseline, `results/baselines/<dataset>/{defaults,random_search,optuna_tpe,flaml}/seed_<k>/`, each with `report.txt`, `result.json`, `run.log` (plus `model.pkl`, `optuna.db` / `flaml.log`).
- Random Search and Optuna evaluate one configuration at a time, so at the same budget they take longer than EvoSynth, which evaluates in parallel (measured for 500 evaluations on Covertype with 10,000 search rows: Random Search 42 min, Optuna 50 min, EvoSynth 12 min; on Adult, where XGBoost evaluations are about a second, Optuna 3.3 min and EvoSynth 3.0 min).

### 3.4 Ablations

```bash
evosynth ablation --setup CVGS-500                                   # pop 40, 2 workers (from [experiments.ablation])
evosynth ablation --setup CVGS-500 --max-workers 8 --subsample 10000 --splits stratified,random,elite
```

- **Outputs:** `results/ablation/<dataset>/cvgs_500/seed_<k>/` with `summary.txt` (including a generated conclusion), `individual_runs/`, `isolated_population.csv`, `population[_<split>].csv` (EvoSynth), `checkpoints/`, `result.json`.
- **Resuming:** re-run the same command. Completed parts are skipped and an interrupted optimizer continues exactly from its last checkpoint, with its full internal state; `run.log` is appended to.

### 3.5 Vision NAS and the speed-up benchmark

```bash
evosynth nas                   # CNN NAS on Fashion-MNIST: 160 evals, pop 40, 2 epochs per candidate (~5 s each)
evosynth speedup               # parallel threads and early stopping, measured on Covertype
```

### 3.6 Experiment plans and summary tables

```bash
py -3.14 tools/run_experiments.py smoke              # the Phase 5 plan; restartable
py -3.14 tools/run_experiments.py core --dry-run     # list a larger plan's jobs
evosynth summary                                     # results/summary/: tables over seeds, convergence curves
```

See [07-experiments.md](07-experiments.md) §8.

### 3.7 Inspecting settings

```bash
evosynth settings show engine                         # one section, resolved
evosynth settings show --preset search_full --set engine.budget=600
```

## 4. Files written by any run

Folder: `results/<experiment>/<dataset>/<variant>/seed_<run.seed>/`

| File | Content |
|---|---|
| `manifest.json` | Command, every resolved setting, git commit, library versions, start/end time, status |
| `result.json` | Final numbers of the run |
| `run.log` | Mirror of everything printed (UTF-8) |
| `events.jsonl` | Engine events (same stream as the dashboard) |
| `population.csv` | Every evaluation of the EvoSynth run: `Eval_ID,Phase,Algorithm,Fitness,<decoded hyperparameters>` |

Only `manifest.json`, `result.json` and `summary/` are committed to git. The panel-review results are kept unchanged in `results/archive/2026-11-19_covertype_v1/`.

## 5. Live dashboard

The engine looks for the server when a run starts, so start the server first.

```bash
# Terminal 1 — telemetry relay on :8000
evosynth serve

# Terminal 2 — dashboard on :5173
cd dashboard
npm install
npm run dev

# Terminal 3 — start any experiment
evosynth run
```

Open `http://localhost:5173`. If the server was not running when the experiment started, the run still completes and its events are still saved to `events.jsonl`. Dashboard scripts:
- `npm run build` for a production build;
- `npm run preview` to serve that build;
- `npm run lint` to run oxlint.

## 6. Documentation site

```bash
cd docs
npm install
npm run docs:dev        # local preview (VitePress)
```

Pushing changes under `docs/` to `main` or `PostReview2` triggers `.github/workflows/deploy-docs.yml`, which builds and publishes to GitHub Pages. The handbook and the panel-defense guide are not part of the site; they live in the top-level `handbook/` folder.

## 7. Using EvoSynth as a library

```python
from evosynth.config import load_config, set_settings
from evosynth.core.decoding import decode_vector_to_dict
from evosynth.engine.controller import DynamicOptimizer
from evosynth.evaluators.tabular import TabularEvaluator
from evosynth.search_spaces.tabular import get_tabular_search_space

set_settings(load_config(overrides=["bandit.c=0.1"]))     # optional; defaults to settings/

space = get_tabular_search_space(include_model_type=True, include_fe_strategy=True)
evaluator = TabularEvaluator("covertype", subsample_size=20000)

opt = DynamicOptimizer(search_space=space, evaluator=evaluator,
                       budget=300, pop_size=30, t1_iters=9, k_iters=6)   # others from settings
best = opt.run()                                  # Individual
print(best.fitness, best.source_algorithm)        # e.g. 86.4  'T2_CMAES'
print(decode_vector_to_dict(best.vector, space))  # dict of real hyperparameters
print(opt.history[-1])                            # {'eval_id', 'fitness', 'best_so_far', 'algo'}
```

Used this way, nothing is written to disk. Wrap the code in `evosynth.experiments.run_context.RunContext(...)` to get a run folder with `manifest.json`, `run.log` and `events.jsonl`.

**Choosing settings:**
- Keep `pop_size` at least 20 so each Tier 1 island has at least 2 members (checked by `config.validate`). Use 40 or more for islands of 4 or more, which gives more meaningful diversity scores.
- Make sure `budget` comfortably exceeds `pop_size + t1_iters·4·(0.1·pop_size)`. Otherwise Tier 2 has no budget left (the run prints a warning).
- `pop_size` must be at least 40, so each Tier 1 island has at least 4 members.
- `k_iters` is lowered automatically (`engine.auto_k`) so that Tier 2 reaches at least two checkpoints, where SWAPs, injections and elite sharing happen.
- For your own data, write an evaluator as described in [05](05-search-space-and-evaluators.md) §6. For a new optimizer, see [04](04-optimizers.md) §7.

## 8. Tests

`python -m pytest` (needs the `dev` extra) runs the 88 tests in `tests/` in about 20 seconds, on the toy objective and small synthetic data: settings and validation; search spaces and decoding; seeding and reproducibility; the engine (bandit, metrics, tier split, auto k, inject, CMA-ES updates); PSO and DE behaviour; datasets, splits, metrics, early stopping, failed folds and confirm-before-select; the ablation (resume, passive scoring, generated conclusion), refit, baselines and summary tables; run folders and the `evosynth` command.

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `evosynth` is not recognised | Its scripts folder is not on `PATH`; use `python -m evosynth`. |
| `evosynth: settings error: ...` | A value in `settings/` or a `--set` override is invalid; the message names it. `evosynth settings check` validates the folder. |
| Dashboard stays on "Waiting for engine start…" | Start `evosynth serve`; the engine re-checks every 10 s, so the dashboard connects within seconds. Check that `http://localhost:8000/docs` loads. |
| `[EVAL] fold … failed` lines | A configuration could not be trained (it scores 0 for that fold). Occasional failures are normal; many suggest a data or settings problem. The count is in `result.json` (`failed_folds`). |
| `matplotlib is not a writable directory` warning (Windows) | Harmless. Set `MPLCONFIGDIR` to a writable folder to silence it. |
| Optuna or FLAML not found | `pip install -e ".[baselines]"`. |
| A run interrupted midway | `evosynth scale` and `evosynth ablation` checkpoint; the ablation and Optuna resume automatically when re-run. The manifest status says `interrupted`. |
| Out of memory on long ablations | Lower `--max-workers` (the ablation preset uses 2). |
| `budget … leaves about N Tier 2 iteration(s)` warning | Too little budget after Tier 1. Raise `budget` or lower `t1_iters` / `pop_size`. |

---

## Quiz

**1.** Which command installs EvoSynth with every extra?
<details><summary>Answer</summary>`pip install -e ".[all]"`.</details>

**2.** How do you download all datasets in advance, and where do they go?
<details><summary>Answer</summary>`evosynth datasets download`; into `.data/` (git-ignored).</details>

**3.** Write the command for a 500-evaluation search on HIGGS with seed 7.
<details><summary>Answer</summary>`evosynth run --full --dataset higgs --seed 7`.</details>

**4.** How would you change the exploration constant for one run only?
<details><summary>Answer</summary>`--set bandit.c=0.2` on the command.</details>

**5.** Which budgets does `evosynth scale` accept?
<details><summary>Answer</summary>300, 500 and 1000 (`experiments.scaling.budgets`).</details>

**6.** What is the smallest allowed population, and why?
<details><summary>Answer</summary>40, so each of the four Tier 1 islands has at least 4 members (`engine.min_island_size`).</details>

**7.** How do you resume an interrupted experiment plan?
<details><summary>Answer</summary>Run the same `tools/run_experiments.py` command again; completed runs are skipped.</details>

**8.** What does a `[EVAL] fold … failed` line mean?
<details><summary>Answer</summary>A configuration could not be trained; that fold scores 0 and the failure is counted.</details>

**9.** How long do the tests take, and do they train real models on real data?
<details><summary>Answer</summary>About 20 seconds; they use the toy objective and small synthetic data.</details>
