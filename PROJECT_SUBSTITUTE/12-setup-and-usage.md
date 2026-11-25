# 12 — Setup & Usage

## 1. Prerequisites

| Component | Version |
|---|---|
| Python | 3.9+. The project was developed and run on Python 3.14 (Windows); the bytecode caches are `cpython-314`. |
| Node.js | 18+ for the dashboard; 20 for the docs CI |
| OS | Developed on Windows (PowerShell orchestrator). The Python code is OS-agnostic. |
| Hardware | CPU only. Reference machine: Intel i5-12450HX (8 cores / 12 threads), 16 GB RAM. |

## 2. Installation

```bash
git clone https://github.com/paarthureddy/EvoSynth.git
cd EvoSynth
python -m venv venv
venv\Scripts\activate            # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

`requirements.txt` contents and what each package is for:

| Package | Purpose |
|---|---|
| `ConfigSpace` | Search-space definitions and vector ↔ configuration decoding |
| `numpy`, `pandas` | Numerics and dataset loading |
| `scikit-learn` | Datasets (Covertype, digits, breast cancer), CV, RandomForest and ExtraTrees |
| `xgboost`, `lightgbm` | Gradient-boosted models |
| `cma` | CMA-ES implementation |
| `torch`, `torchvision` | CNN NAS |
| `fastapi`, `uvicorn` | SSE telemetry server |
| `requests` | Engine → server event posts |
| `optuna`, `flaml` | Competitor baselines |
| `deap`, `matplotlib` | Listed dependencies; no Python file in the repo imports them |
| `reportlab` | Needed only by `scripts/generate_hybrid_workflow_pdf.py`; not in requirements |

**Check the install:**

```bash
python -c "from automl_engine.core.search_space import get_xgboost_search_space; from automl_engine.core.evaluator import XGBoostEvaluator; print('EvoSynth Backend Ready!')"
```

**Dataset.** Covertype is downloaded automatically by `sklearn.datasets.fetch_covtype` on first use and cached in `<repo>/.data/covertype/` (`samples_py3` is about 14 MB). The other datasets ship with scikit-learn or are generated synthetically.

## 3. Running EvoSynth

Run all commands from the repo root. Root wrappers and `scripts/` paths are equivalent.

### 3.1 Covertype benchmark: the headline demo

```bash
python run_covertype.py                 # 120 evals, pop 20, t1=3, k=2  (~10 min incl. 60k/20k refit)
python run_covertype.py --full          # 500 evals, pop 40, t1=5, k=3
python run_covertype.py --budget 200 --pop_size 30 --t1_iters 4 --k_iters 2 --subsample 20000
```

**Output:**
- Steps 0–6 are printed, ending with a summary box: search CV, test accuracy, architecture and FE strategy.
- For reference, the 120-evaluation run on the machine above took 9 min 44 s and gave 86.39% CV / 93.47% test.

### 3.2 Covertype at scale, with checkpointing

```bash
python scripts/run_covertype_evosynth.py --budget 300 --pop_size 30 --checkpoint_interval 10 --run_name Test1_19_Nov
python scripts/run_covertype_evosynth.py --budget 500 --pop_size 30 --checkpoint_interval 10 --run_name Test1_19_Nov
```

- `--budget` must be one of 120, 300, 500 or 1000.
- **Outputs:**
  - `results_forest/<run>/evosynth/run_<budget>.txt`, the full console log;
  - `checkpoints/checkpoint_<budget>.json`, updated every 10 evaluations and finalised with the test accuracy.
- **Typical times:** about 31 min (300) and about 37 min (500).

### 3.3 Baselines (same 60k/20k split)

```bash
python scripts/run_baselines_forest.py --test all --run_name Test1_19_Nov
python scripts/run_baselines_forest.py --test optuna --trials 120        # resumes from its SQLite DB
python scripts/run_baselines_forest.py --test flaml --time_budget 585
```

- `--test` accepts `all`, `extratrees`, `random`, `optuna` or `flaml`. `--output_dir` overrides the folder.
- **Outputs:** `results_forest/<run>/baselines/*.txt` (plus `optuna_covertype.db` and `flaml_run.log`).
- **Times:** ExtraTrees about 1 s; Random Search about 34 min; Optuna about 36 min; FLAML about 10 min.

### 3.4 Ablations

```bash
# Pathological noisy dataset (results/)
python scripts/run_ablation_study.py --setup 120                      # UPGS-120 (~15 min)
python scripts/run_ablation_study.py --setup 2500                     # UPGS-2500 (fast_mode, ~2.2 h)
python scripts/run_ablation_study.py --setup both --dataset pathological --max_workers 8
python scripts/run_ablation_study.py --setup 120 --dataset breast_cancer

# Covertype (results_forest/<run>/ablation/), resumable after interruption
python scripts/run_ablation_study_forest.py --setup CVGS-300 --pop_size 30 --max_workers 2 --checkpoint_interval 10 --run_name Test1_19_Nov
python scripts/run_ablation_study_forest.py --setup CVGS-500 --pop_size 30 --max_workers 2 --checkpoint_interval 10 --run_name Test1_19_Nov
```

- **Resuming CVGS runs:** re-run the same command. Completed optimizers are skipped and interrupted ones continue from their last 10-evaluation checkpoint.
- **Time:** CVGS-300 took about 25–37 min per optimizer, so plan for an overnight run.

### 3.5 Pathological demo and vision NAS

```bash
python main.py                 # 120 evals, pop 20 on the 30% label-noise dataset (3-fold CV)
python main.py --full          # 2500 evals, pop 100, t1=10, k=5
python run_vision_nas.py       # CNN NAS on digits: 80 evals, pop 20, 2 epochs per candidate (~15–20 s)
```

### 3.6 Staged orchestrator (Windows PowerShell)

```powershell
.\run_experiments.ps1                         # interactive menu: 1/2/3/4/A/N/Q
.\run_experiments.ps1 -Stage all              # baselines → EvoSynth 300/500 → CVGS-300 → CVGS-500
.\run_experiments.ps1 -Stage 2 -RunName Test2_25_Nov
```

It writes everything to `results_forest\Test<N>_<dd_MMM>\{baselines,evosynth,ablation\cvgs_300,ablation\cvgs_500,summary}`. See [07-experiments.md](07-experiments.md) §8.

### 3.7 Checking the released dataset

```bash
python scripts/load_dataset.py
```

This loads `pathological_classification_benchmark.csv` and `evosynth_hpo_trajectories.csv` from `data/` or `dataset_release/data/`. Both files come from the Zenodo release `doi:10.5281/zenodo.23645119`. The script prints shapes, class balance, the algorithms and model types present, and the best fitness.

## 4. Files written by any run

| Path | Content |
|---|---|
| `results/run_output.txt` | Mirror of everything printed. Overwritten on each run. |
| `results/csv/warmup_population.csv` | Every Init evaluation: `Eval_ID,Algorithm,Fitness,<decoded hyperparameters>` |
| `results/csv/tier1_population.csv` | Every T1_* evaluation |
| `results/csv/tier2_population.csv` | Every T2_* evaluation |

## 5. Live dashboard

The order matters: the engine checks for the server only when it is imported.

```bash
# Terminal 1 — telemetry relay on :8000
python api_server.py

# Terminal 2 — dashboard on :5173
cd dashboard
npm install
npm run dev

# Terminal 3 — start any experiment
python run_covertype.py
```

Open `http://localhost:5173`. If the server was not running when the experiment started, the run still completes, but no telemetry is sent. Dashboard scripts:
- `npm run build` for a production build;
- `npm run preview` to serve that build;
- `npm run lint` to run oxlint.

## 6. Documentation site

```bash
cd docs
npm install
npm run docs:dev        # local preview (VitePress)
```

Pushing changes under `docs/` to `main` or `PostReview2` triggers `.github/workflows/deploy-docs.yml`, which builds and publishes to GitHub Pages.

To regenerate the workflow PDF: `pip install reportlab`, then `python scripts/generate_hybrid_workflow_pdf.py`. It writes `Hybrid Workflow.pdf` to the repo root.

## 7. Using EvoSynth as a library

```python
from automl_engine.core.search_space import get_xgboost_search_space
from automl_engine.core.covtype_evaluator import CovertypeEvaluator
from automl_engine.engine.controller import DynamicOptimizer, _decode_individual

space = get_xgboost_search_space(include_model_type=True, include_fe_strategy=True)
evaluator = CovertypeEvaluator(subsample_size=20000)

opt = DynamicOptimizer(search_space=space, evaluator=evaluator,
                       budget=300, pop_size=30, t1_iters=9, k_iters=6,
                       swap_threshold=0.10, stagnation_limit=3,
                       elite_pct=0.20, inject_pct=0.20, max_workers=8)
best = opt.run()                                  # Individual
print(best.fitness, best.source_algorithm)        # e.g. 86.4  'T2_CMAES'
print(_decode_individual(best, space))            # dict of real hyperparameters
print(opt.history[-1])                            # {'eval_id', 'fitness', 'best_so_far', 'algo'}
```

**Choosing settings:**
- Keep `pop_size` at least 20 so each Tier 1 island has at least 2 members. Use 40 or more for islands of 4 or more, which gives more meaningful diversity scores.
- Make sure `budget` comfortably exceeds `pop_size + t1_iters·4·(0.1·pop_size)`. Otherwise Tier 2 has no budget left.
- To see SWAPs, injections and elite sharing in Tier 2, choose `k_iters` small enough that Tier 2 reaches at least one checkpoint.
- For your own data, write an evaluator as described in [05](05-search-space-and-evaluators.md) §6. For a new optimizer, see [04](04-optimizers.md) §7.

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Dashboard stays on "Waiting for engine start…" | Start `api_server.py` **before** the experiment. Check that `http://localhost:8000/docs` loads. |
| Many `use_label_encoder` warnings | Harmless. Newer XGBoost ignores that argument in the Covertype evaluator. |
| `matplotlib is not a writable directory` warning (Windows) | Harmless. Set `MPLCONFIGDIR` to a writable folder to silence it. |
| Optuna or FLAML not found | `pip install optuna flaml`. The orchestrator installs them automatically. |
| A run interrupted midway | Use the CVGS scripts or `run_covertype_evosynth.py`, which checkpoint. The CVGS ablation resumes automatically. |
| Out of memory on long ablations | Lower `--max_workers` (the orchestrator uses 2 for CVGS). |
| Tier 2 shows no swaps or sharing | `k_iters` is too large for the budget left after Tier 1. Lower `k_iters` or raise `budget`. |
