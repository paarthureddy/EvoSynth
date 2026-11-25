# Archive: Covertype experiments of 19 Nov 2026 (v1, panel review)

These are the results presented at the Panel Review. They were produced by the code at git tag
`v1-panel-review` (before the restructure and before the Phase 0–6 fixes), so they **cannot be
reproduced with the current code** and must not be mixed with new runs. They are kept unchanged
as a historical record. The only edit is that `evosynth/run_120.txt` was converted from UTF-16 to UTF-8.

Originally stored in `results_forest/Test1_19_Nov/`, written by the old scripts (`scripts/…`, now the
`evosynth` command).

| Folder | Produced by (old script → current command) | Contents |
|---|---|---|
| `baselines/` | `run_baselines_forest.py` → `evosynth baselines` | Default ExtraTrees, Random Search (120), Optuna TPE (120, + SQLite study), FLAML (585 s) reports |
| `evosynth/` | `run_covertype.py` (120) and `run_covertype_evosynth.py` (300, 500) → `evosynth run` / `evosynth scale` | Full console logs of EvoSynth at budgets 120, 300, 500 |
| `ablation/cvgs_300/` | `run_ablation_study_forest.py` → `evosynth ablation` | Isolated GA / PSO / DE / CMA-ES vs EvoSynth at budget 300: summary table and one report per optimizer |
| `ablation/cvgs_500/` | same | Budget 500 – **incomplete**: the four isolated optimizers finished, the EvoSynth run did not (its checkpoint state says "running"). The summary's EvoSynth row (86.59 %) matches the budget-500 scaling run in `evosynth/`, and there are no per-optimizer reports |
| `summary/` | written by hand | `main_comparison_table.txt` (the table shown at review), `convergence_curve.csv` |

Known caveats (fixed in later phases): the ablation summaries' EvoSynth diversity / composite / UCB
values were hard-coded constants; methods were refit with different protocols; only one seed was run.

Kept only on the local machine (git-ignored): `checkpoints/` (resume state) and `ablation/*/csv/`
(per-evaluation rows).
