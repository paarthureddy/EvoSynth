# EvoSynth settings

Every tunable value in the project lives in this folder. Change a value, re-run, done:
no Python file holds its own copy of a setting.

## Which file holds what

| File | Sections | What you tune there |
|---|---|---|
| [run.toml](run.toml) | `[run]` | Random state used by every split/model, number of parallel evaluation threads |
| [engine.toml](engine.toml) | `[engine]`, `[bandit]`, `[composite]`, `[safeguards]` | Budget, population, Tier 1 iterations, checkpoint interval `k`, Tier 1 share; UCB exploration constant `c`; the five composite-score weights; SWAP margin, grace period, cooldown, stagnation limit, injection and elite-sharing percentages |
| [optimizers.toml](optimizers.toml) | `[optimizers.ga]`, `.pso`, `.de`, `.cmaes` | GA mutation rate/σ/tournament size; PSO `w`, `c1`, `c2`, initial velocity; DE `F`, `CR`; CMA-ES σ0, minimum population, default mean |
| [search_spaces.toml](search_spaces.toml) | `[search_space.tabular]`, `.nas`, `.toy` | Model families, feature-engineering strategies and every hyperparameter range (tabular); CNN channels/units/dropout/LR/optimizers/activations (NAS); toy test space |
| [evaluation.toml](evaluation.toml) | `[evaluation]`, `[features]`, `[refit]` | CV folds, early stopping, search-time tree cap, RF/ET depth rule, failed-fold score; feature-engineering detection thresholds; refit train/test rows and minimum tree counts |
| [datasets.toml](datasets.toml) | `[datasets.<name>]` | One section per dataset: source, search subsample size, label offset, metric, XGBoost eval metric |
| [telemetry.toml](telemetry.toml) | `[telemetry]` | Dashboard server host/port, the address the engine reports to, timeouts |
| [experiments.toml](experiments.toml) | `[experiments.<name>]` | Presets for each command: `search_demo`, `search_full` (`evosynth run`), `scaling` (`evosynth scale`), `ablation` (+ `passive_telemetry`), `baselines`, `nas` |

Inside every file, each setting has a comment saying **what it controls**, **where it is used**
(`Used in: <file> → <function>`) and its **valid range**.

## Rules

- All `*.toml` files in this folder are merged into one set of settings. A top-level section
  (e.g. `[engine]`) may be defined in **one file only**; defining it twice is an error.
- Values are checked when loaded (`src/evosynth/config.py → validate`): e.g. composite weights
  must sum to 1.0, percentages must be in range, Tier 1 islands need at least 2 members,
  budget must exceed population. Invalid settings stop the run with a clear message.

## How a value is chosen for a run (later wins)

1. `settings/*.toml` (this folder)
2. the preset of the command, e.g. `[experiments.search_demo]` for `evosynth run`
3. command-line flags, e.g. `evosynth run --budget 300`
4. `--set` overrides, e.g. `evosynth run --set bandit.c=0.2 --set optimizers.pso.w=0.7`

Each run records the fully resolved settings in its `manifest.json`
(`results/<experiment>/<dataset>/<variant>/seed_<random_state>/`), so results always say which
values produced them.

## Useful commands

```bash
py -3.14 -m evosynth settings check                 # validate the folder
py -3.14 -m evosynth settings show engine           # print one section (resolved)
py -3.14 -m evosynth settings show --preset search_full --set engine.budget=600
py -3.14 -m evosynth run --set composite.fitness=0.4 --set composite.speed=0.15   # one-off change
```

Use another settings folder (e.g. a copy for an experiment) with `--settings PATH`.
