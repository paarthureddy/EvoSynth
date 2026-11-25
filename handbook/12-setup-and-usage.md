# 12 — Setup & Usage

## 1. Prerequisites

| Component | Version |
|---|---|
| Python | 3.11+ (`requires-python` in `pyproject.toml`). Developed and run on Python 3.14 (Windows). |
| Node.js | 18+ for the dashboard; 20 for the docs CI |
| OS | Developed on Windows (PowerShell orchestrator). The Python code is OS-agnostic. |
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

**Datasets.** Each dataset in `settings/datasets.toml` is downloaded on first use into `<repo>/.data/` (git-ignored). Covertype comes from `sklearn.datasets.fetch_covtype` (about 14 MB).

## 3. Running EvoSynth

Run all commands from the repo root. Every command accepts:
- `--set SECTION.KEY=VALUE` (repeatable) to override any setting for this run, e.g. `--set bandit.c=0.2`;
- `--settings PATH` to use another settings folder.

Experiment commands also accept `--results-dir PATH` (write under PATH instead of `results/`); `run`, `scale` and `nas` accept `--variant NAME` (name of the run folder). Each run writes to `results/<experiment>/<dataset>/<variant>/seed_<random_state>/` (see §4).

### 3.1 Covertype benchmark: the headline demo

```bash
evosynth run                      # preset search_demo: 120 evals, pop 20, t1=3, k=2 (+ 60k/20k refit)
evosynth run --full               # preset search_full: 500 evals, pop 40, t1=5, k=3
evosynth run --budget 200 --pop-size 30 --t1-iters 4 --k-iters 2 --subsample 20000
evosynth run --no-refit           # search only
```

**Output** (`results/search/covertype/search_demo/seed_42/`): Steps 0–6 are printed, ending with a summary box: search CV, test accuracy, architecture and FE strategy. For reference, the v1 120-evaluation run on the machine above took 9 min 44 s and gave 86.39% CV / 93.47% test.

### 3.2 Covertype at scale, with checkpointing

```bash
evosynth scale --budget 300
evosynth scale --budget 500 --pop-size 30 --checkpoint-interval 10
```

- `--budget` must be one of `experiments.scaling.budgets` (120, 300, 500, 1000). `t1_iters` and `k_iters` follow the schedule in `[experiments.scaling]`.
- **Outputs** (`results/scaling/covertype/budget_<N>/seed_42/`): `run.log`; `checkpoint.json`, updated every 10 evaluations and finalised with the test accuracy; `result.json`.
- **Typical times (v1):** about 31 min (300) and about 37 min (500).

### 3.3 Baselines (same 60k/20k split)

```bash
evosynth baselines --test all
evosynth baselines --test optuna --trials 120        # resumes from its SQLite study
evosynth baselines --test flaml --time-budget 585
```

- `--test` accepts `all`, `extratrees`, `random`, `optuna` or `flaml`.
- **Outputs:** one folder per baseline, `results/baselines/covertype/{default_extratrees,random_search,optuna_tpe,flaml}/seed_42/`, each with `report.txt`, `result.json`, `run.log` (plus `optuna.db` / `flaml.log`).
- **Times (v1):** ExtraTrees about 1 s; Random Search about 34 min; Optuna about 36 min; FLAML about 10 min.

### 3.4 Ablations

```bash
evosynth ablation --setup CVGS-300                        # pop 30, 2 workers (from [experiments.ablation])
evosynth ablation --setup CVGS-500 --max-workers 2 --checkpoint-interval 10
```

- **Outputs:** `results/ablation/covertype/cvgs_300/seed_42/` with `summary.txt`, `individual_runs/`, `isolated_population.csv`, `population.csv` (EvoSynth), `checkpoints/`, `result.json`.
- **Resuming:** re-run the same command. Completed optimizers are skipped and interrupted ones continue from their last 10-evaluation checkpoint; `run.log` is appended to.
- **Time (v1):** CVGS-300 took about 25–37 min per optimizer, so plan for an overnight run.

### 3.5 Vision NAS

```bash
evosynth nas                   # CNN NAS on digits: 80 evals, pop 20, 2 epochs per candidate (~15–20 s)
```

### 3.6 Staged orchestrator (Windows PowerShell)

```powershell
.\tools\run_experiments.ps1                    # interactive menu: 1/2/3/4/A/Q
.\tools\run_experiments.ps1 -Stage all         # baselines → EvoSynth 300/500 → CVGS-300 → CVGS-500
.\tools\run_experiments.ps1 -Stage 2 -Extra "--set","run.random_state=7"
```

It calls the commands above, so everything lands in the usual run folders. See [07-experiments.md](07-experiments.md) §8.

### 3.7 Inspecting settings

```bash
evosynth settings show engine                         # one section, resolved
evosynth settings show --preset search_full --set engine.budget=600
```

## 4. Files written by any run

Folder: `results/<experiment>/<dataset>/<variant>/seed_<random_state>/`

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
- Make sure `budget` comfortably exceeds `pop_size + t1_iters·4·(0.1·pop_size)`. Otherwise Tier 2 has no budget left.
- To see SWAPs, injections and elite sharing in Tier 2, choose `k_iters` small enough that Tier 2 reaches at least one checkpoint.
- For your own data, write an evaluator as described in [05](05-search-space-and-evaluators.md) §6. For a new optimizer, see [04](04-optimizers.md) §7.

## 8. Tests

`python -m pytest` (needs the `dev` extra) runs the smoke tests in `tests/` in about 5 seconds, without training any model: settings loading, presets, overrides and validation; search spaces and decoding; a full `DynamicOptimizer` run on the toy objective; run folders (`manifest.json`, `result.json`, `run.log`, `events.jsonl`); and the `evosynth` command.

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `evosynth` is not recognised | Its scripts folder is not on `PATH`; use `python -m evosynth`. |
| `evosynth: settings error: ...` | A value in `settings/` or a `--set` override is invalid; the message names it. `evosynth settings check` validates the folder. |
| Dashboard stays on "Waiting for engine start…" | Start `evosynth serve` **before** the experiment. Check that `http://localhost:8000/docs` loads. |
| Many `use_label_encoder` warnings | Harmless. Newer XGBoost ignores that argument (removed in Phase 3). |
| `matplotlib is not a writable directory` warning (Windows) | Harmless. Set `MPLCONFIGDIR` to a writable folder to silence it. |
| Optuna or FLAML not found | `pip install -e ".[baselines]"`. The orchestrator installs them automatically. |
| A run interrupted midway | `evosynth scale` and `evosynth ablation` checkpoint; the ablation and Optuna resume automatically when re-run. The manifest status says `interrupted`. |
| Out of memory on long ablations | Lower `--max-workers` (the ablation preset uses 2). |
| Tier 2 shows no swaps or sharing | `k_iters` is too large for the budget left after Tier 1. Lower `k_iters` or raise `budget`. |
