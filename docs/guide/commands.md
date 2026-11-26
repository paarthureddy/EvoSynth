# Commands

Every command in the EvoSynth project, with its options and defaults. All experiment commands are subcommands of `evosynth` (installed by `pip install -e .`). If `evosynth` is not on your `PATH`, use `python -m evosynth` instead (on Windows with several Pythons: `py -3.14 -m evosynth`).

Defaults come from the `settings/` folder; the values shown are the shipped defaults.

---

## 1. Setup

```bash
git clone https://github.com/paarthureddy/EvoSynth.git
cd EvoSynth
python -m venv venv
venv\Scripts\activate              # Windows
source venv/bin/activate           # Linux / macOS
pip install -e ".[all]"            # package + every extra (NAS, baselines, tests)
pip install -e .                   # core only
pip install -e ".[baselines]"      # adds Optuna and FLAML
pip install -e ".[nas]"            # adds PyTorch and torchvision
```

```bash
evosynth datasets list             # registered datasets, metric, cached or not
evosynth datasets download         # download all 8 datasets into .data/ (about 230 MB)
evosynth datasets download adult higgs   # only some
evosynth settings check            # validate the settings/ folder
```

---

## 2. Options every command accepts

| Option | Meaning |
|---|---|
| `--set SECTION.KEY=VALUE` | Override one setting for this run; repeatable. Example: `--set bandit.c=0.2 --set composite.fitness=0.4 --set composite.speed=0.15` |
| `--settings PATH` | Use another settings folder or a single `.toml` file |

Experiment commands (`run`, `scale`, `ablation`, `baselines`, `nas`, `speedup`) also accept:

| Option | Meaning |
|---|---|
| `--seed N` | Search seed for this run (sets `run.seed`, default 42) |
| `--seeds 1,2,3` | Repeat the command once per seed; one `seed_<k>` folder each (not combinable with `--seed`) |
| `--results-dir PATH` | Write under `PATH` instead of `results/` |
| `--variant NAME` | Name of the run-folder level (`run`, `scale`, `nas`, `speedup`) |
| `--subsample N` | Search on a stratified N-row subset of the dataset's search rows (`run`, `scale`, `ablation`, `baselines`, `speedup`) |

Every run writes to `results/<experiment>/<dataset>/<variant>/seed_<seed>/`.

---

## 3. Experiment commands

### `evosynth run`: one search, confirmation and refit

| Option | Default | Meaning |
|---|---|---|
| `--preset NAME` | `search_demo` | `[experiments.<NAME>]` to apply: `search_demo` (budget 200, pop 40, t1 3, k 2) or `search_full` |
| `--full` | | Shortcut for `--preset search_full` (budget 500, pop 40, t1 5, k 3) |
| `--dataset NAME` | `covertype` | `covertype`, `higgs`, `adult`, `airlines`, `poker_hand`, `miniboone`, `jannis` |
| `--budget N` | from preset | Total evaluations |
| `--pop-size N` | from preset | Initial population (at least 40) |
| `--t1-iters N` | from preset | Tier 1 iterations |
| `--k-iters N` | from preset | Checkpoint interval (upper bound; lowered automatically) |
| `--subsample N` | 20,000 | Search rows |
| `--no-refit` | | Stop after confirm-before-select |

```bash
evosynth run
evosynth run --full --dataset adult --seed 7
evosynth run --budget 300 --t1-iters 4 --subsample 10000
evosynth run --full --seeds 42,43,44
```

### `evosynth scale`: budget scaling with checkpoints

| Option | Default | Meaning |
|---|---|---|
| `--budget N` | 500 | One of 300, 500, 1000 |
| `--pop-size N` | 40 | Initial population |
| `--checkpoint-interval N` | 10 | Write `checkpoint.json` every N evaluations |
| `--dataset NAME` | `covertype` | |

```bash
evosynth scale
evosynth scale --budget 1000 --dataset higgs
```

### `evosynth baselines`: reference methods

| Option | Default | Meaning |
|---|---|---|
| `--test` | `all` | `all`, `defaults` (untuned XGBoost / LightGBM / RandomForest / ExtraTrees), `random`, `optuna`, `flaml` |
| `--trials N` | 120 | Random Search / Optuna evaluations (match the EvoSynth budget) |
| `--time-budget S` | 585 | FLAML seconds (match the EvoSynth wall time) |
| `--dataset NAME` | `covertype` | |

```bash
evosynth baselines --test all
evosynth baselines --test random --trials 500 --subsample 10000
evosynth baselines --test optuna --trials 500        # resumes from its SQLite study
evosynth baselines --test flaml --time-budget 900
evosynth baselines --test defaults --dataset adult
```

### `evosynth ablation`: each optimizer alone vs EvoSynth

| Option | Default | Meaning |
|---|---|---|
| `--setup NAME` | `CVGS-500` | `CVGS-500` (budget 500) or `CVGS-1000` (budget 1000) |
| `--splits LIST` | `stratified` | EvoSynth tier splits to run; `stratified,random,elite` is the split ablation |
| `--pop-size N` | 40 | |
| `--max-workers N` | 2 | Parallel evaluation threads (raise if memory allows) |
| `--checkpoint-interval N` | 10 | Save resume state every N evaluations |
| `--dataset NAME` | `covertype` | |

```bash
evosynth ablation --setup CVGS-500
evosynth ablation --setup CVGS-500 --max-workers 8 --subsample 10000 --splits stratified,random,elite
```

Re-running the same command resumes an interrupted ablation.

### `evosynth nas`: CNN architecture search

Fashion-MNIST, budget 160, pop 40, 2 epochs per candidate (`[experiments.nas]`).

```bash
evosynth nas
evosynth nas --seed 7 --set experiments.nas.epochs=3
```

### `evosynth speedup`: measure parallel threads and early stopping

| Option | Default | Meaning |
|---|---|---|
| `--dataset NAME` | `covertype` | |
| `--configurations N` | 40 | Configurations evaluated under each of the 4 conditions |
| `--workers N` | all CPU cores | Parallel threads |

```bash
evosynth speedup
evosynth speedup --subsample 10000 --workers 8
```

### `evosynth summary`: tables and curves from finished runs

| Option | Default | Meaning |
|---|---|---|
| `--results-dir PATH` | `results/` | Folder to read; writes `PATH/summary/` |

```bash
evosynth summary
```

---

## 4. Other commands

```bash
evosynth serve                         # telemetry relay for the live dashboard (http://localhost:8000)
evosynth serve --host 0.0.0.0 --port 8001

evosynth settings show                 # every resolved setting (JSON)
evosynth settings show engine          # one section
evosynth settings show optimizers.pso
evosynth settings show --preset search_full --set engine.budget=600
evosynth settings check

evosynth --help                        # list of commands
evosynth run --help                    # options of one command
```

---

## 5. Experiment runner

```bash
py -3.14 tools/run_experiments.py smoke            # seed 42; Covertype + Adult; ablation, speed-up, NAS
py -3.14 tools/run_experiments.py lean             # seeds 42-44; 4 datasets; split ablation
py -3.14 tools/run_experiments.py core             # seeds 42-44; all 7 tabular datasets
py -3.14 tools/run_experiments.py core --dry-run   # list jobs and which are done
```

Restart the same command to continue after an interruption; progress is in `results/runner.log`.

```bash
py -3.14 tools/export_splits.py OUT_DIR            # export the exact dataset splits (EvoSynth-Bench)
```

---

## 6. Dashboard and documentation site

```bash
cd dashboard
npm install
npm run dev                            # http://localhost:5173 (start `evosynth serve` too)

cd docs
npm install
npm run docs:dev                       # documentation site, locally
npm run docs:build                     # static build
```

---

## 7. Tests

```bash
pip install -e ".[dev]"
python -m pytest                       # 88 tests, about 20 seconds
python -m pytest tests/test_engine_phase1.py -q
```

---

## 8. Recipes

| Goal | Command |
|---|---|
| Quick demo with live dashboard | `evosynth serve`, `npm run dev` in `dashboard/`, then `evosynth run` |
| Full comparison on one dataset, seed 42 | `evosynth run --full --dataset adult --subsample 10000`, then `evosynth baselines --test all --dataset adult --trials 500 --subsample 10000` |
| Same, three seeds | add `--seeds 42,43,44` to both |
| Try a different exploration constant | `evosynth run --set bandit.c=0.3` |
| Turn off confirm-before-select | `evosynth run --set selection.enabled=false` |
| Turn off early stopping | `evosynth run --set evaluation.early_stopping=false` |
| Use the v1 tier split | `evosynth run --set engine.tier_split=elite` |
| Rebuild all tables after new runs | `evosynth summary` |
