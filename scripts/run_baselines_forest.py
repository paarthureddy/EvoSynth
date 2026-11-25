"""
Competitor & Floor Baselines Runner — UCI Forest Covertype
===========================================================
Runs and logs the baseline comparisons on the exact same 60k train / 20k test split:
  1. Default ExtraTrees Classifier (Untuned Floor)
  2. Random Search (120 evaluations across the 9-dimensional search space)
  3. Optuna TPE (120 trials, with SQLite persistence for auto-resume)
  4. FLAML AutoML (Time-budget matched to EvoSynth's ~10 min runtime)

Outputs saved to: results_forest/baselines/
"""

import os
import sys
import time
import json
import argparse
import numpy as np

# Ensure project root is in sys.path
_current = os.path.abspath(os.path.dirname(__file__))
while _current and not os.path.exists(os.path.join(_current, 'automl_engine')):
    _parent = os.path.dirname(_current)
    if _parent == _current:
        break
    _current = _parent
_ROOT_DIR = _current
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)


from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


def load_covertype_splits():
    print("  [DATA] Loading Covertype dataset (60,000 train / 20,000 test)...")
    _data_home = os.path.join(_ROOT_DIR, ".data")
    data = fetch_covtype(data_home=_data_home)
    X = data.data.astype(np.float32)
    y = data.target.astype(int) - 1
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, train_size=60000, test_size=20000, stratify=y, random_state=42
    )
    return X_tr, X_te, y_tr, y_te


def run_default_extratrees(out_dir):
    log_file = os.path.join(out_dir, "default_extratrees.txt")
    print("\n" + "=" * 70)
    print(" RUNNING BASELINE 1: DEFAULT EXTRATREES (UNTUNED FLOOR)")
    print("=" * 70)
    X_tr, X_te, y_tr, y_te = load_covertype_splits()

    start = time.time()
    model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    fit_time = time.time() - start

    preds = model.predict(X_te)
    acc = float(np.mean(preds == y_te)) * 100.0

    print(f"  [RESULT] Default ExtraTrees Accuracy : {acc:.2f}%")
    print(f"  [TIME]   Training Time               : {fit_time:.1f}s")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(" BASELINE REPORT: DEFAULT EXTRATREES (UNTUNED)\n")
        f.write("=" * 70 + "\n")
        f.write(f" Model                : ExtraTreesClassifier (sklearn defaults)\n")
        f.write(f" Train Split          : 60,000 samples\n")
        f.write(f" Test Split           : 20,000 samples (held-out)\n")
        f.write(f" Accuracy             : {acc:.2f}%\n")
        f.write(f" Fit Duration         : {fit_time:.1f} seconds\n")
        f.write("=" * 70 + "\n")
    return acc


def run_random_search(out_dir, n_trials=120):
    log_file = os.path.join(out_dir, "random_search.txt")
    print("\n" + "=" * 70)
    print(f" RUNNING BASELINE 2: RANDOM SEARCH ({n_trials} EVALUATIONS)")
    print("=" * 70)

    from automl_engine.core.search_space import get_xgboost_search_space
    from automl_engine.core.covtype_evaluator import CovertypeEvaluator
    from automl_engine.core.individual import Individual

    space = get_xgboost_search_space(include_model_type=True, include_fe_strategy=True)
    evaluator = CovertypeEvaluator(subsample_size=20000)
    dims = len(list(space.keys()))

    best_acc = 0.0
    best_vec = None
    start = time.time()

    for trial in range(1, n_trials + 1):
        vec = np.random.rand(dims)
        ind = Individual(vec)
        score = evaluator.evaluate(ind)
        if score > best_acc:
            best_acc = score
            best_vec = vec
        if trial % 10 == 0 or trial == n_trials:
            print(f"  Trial {trial:>3}/{n_trials} | Best CV Accuracy: {best_acc:.2f}% | Elapsed: {(time.time()-start):.1f}s")

    # Refit winning vector on 60k/20k
    X_tr, X_te, y_tr, y_te = load_covertype_splits()
    from scripts.run_covertype_evosynth import decode_vector_to_dict
    from automl_engine.core.generic_feature_engineer import GenericFeatureEngineer
    import xgboost as xgb
    import lightgbm as lgb

    best_cfg = decode_vector_to_dict(best_vec, space)
    m_type = best_cfg.get("model_type", "ExtraTrees")
    fe_strat = best_cfg.get("fe_strategy", "none")

    fe = GenericFeatureEngineer(strategy=fe_strat)
    if fe_strat != "none":
        X_tr = fe.fit_transform(X_tr)
        X_te = fe.transform(X_te)

    if m_type == "ExtraTrees":
        model = ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    elif m_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    elif m_type == "LightGBM":
        model = lgb.LGBMClassifier(n_estimators=150, random_state=42, n_jobs=-1, verbose=-1)
    else:
        model = xgb.XGBClassifier(n_estimators=150, random_state=42, n_jobs=-1, eval_metric="mlogloss")

    model.fit(X_tr, y_tr)
    test_acc = float(np.mean(model.predict(X_te) == y_te)) * 100.0

    total_time = time.time() - start
    print(f"  [RESULT] Random Search Test Accuracy: {test_acc:.2f}% (Model: {m_type})")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f" BASELINE REPORT: RANDOM SEARCH ({n_trials} TRIALS)\n")
        f.write("=" * 70 + "\n")
        f.write(f" Search Phase CV Acc  : {best_acc:.2f}%\n")
        f.write(f" Refit Test Accuracy  : {test_acc:.2f}%\n")
        f.write(f" Discovered Model     : {m_type}\n")
        f.write(f" Discovered FE        : {fe_strat}\n")
        f.write(f" Total Runtime        : {total_time/60:.1f} minutes\n")
        f.write("=" * 70 + "\n")
    return test_acc


def run_optuna_baseline(out_dir, n_trials=120):
    log_file = os.path.join(out_dir, "optuna.txt")
    db_file = os.path.join(out_dir, "optuna_covertype.db")
    print("\n" + "=" * 70)
    print(f" RUNNING BASELINE 3: OPTUNA TPE SAMPLER ({n_trials} TRIALS)")
    print("=" * 70)

    try:
        import optuna
    except ImportError:
        print("  [!] Optuna not installed. Please run: pip install optuna")
        return None

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from automl_engine.core.covtype_evaluator import CovertypeEvaluator
    from automl_engine.core.search_space import get_xgboost_search_space
    from automl_engine.core.individual import Individual

    evaluator = CovertypeEvaluator(subsample_size=20000)
    space = get_xgboost_search_space(include_model_type=True, include_fe_strategy=True)
    dims = len(list(space.keys()))

    storage_url = f"sqlite:///{os.path.abspath(db_file)}"
    study = optuna.create_study(
        study_name="covertype_hpo",
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)
    print(f"  [OPTUNA] Prior completed trials: {completed}/{n_trials}. Running remaining: {remaining}...")

    def objective(trial):
        vec = np.array([trial.suggest_float(f"x_{i}", 0.0, 1.0) for i in range(dims)])
        ind = Individual(vec)
        return evaluator.evaluate(ind)

    start = time.time()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    search_time = time.time() - start

    best_cv = study.best_value
    best_vec = np.array([study.best_params[f"x_{i}"] for i in range(dims)])

    from scripts.run_covertype_evosynth import decode_vector_to_dict
    from automl_engine.core.generic_feature_engineer import GenericFeatureEngineer
    import xgboost as xgb
    import lightgbm as lgb

    best_cfg = decode_vector_to_dict(best_vec, space)
    m_type = best_cfg.get("model_type", "ExtraTrees")
    fe_strat = best_cfg.get("fe_strategy", "none")

    X_tr, X_te, y_tr, y_te = load_covertype_splits()
    fe = GenericFeatureEngineer(strategy=fe_strat)
    if fe_strat != "none":
        X_tr = fe.fit_transform(X_tr)
        X_te = fe.transform(X_te)

    if m_type == "ExtraTrees":
        model = ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    elif m_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    elif m_type == "LightGBM":
        model = lgb.LGBMClassifier(n_estimators=150, random_state=42, n_jobs=-1, verbose=-1)
    else:
        model = xgb.XGBClassifier(n_estimators=150, random_state=42, n_jobs=-1, eval_metric="mlogloss")

    model.fit(X_tr, y_tr)
    test_acc = float(np.mean(model.predict(X_te) == y_te)) * 100.0

    print(f"  [RESULT] Optuna TPE Test Accuracy: {test_acc:.2f}% (CV: {best_cv:.2f}%)")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f" BASELINE REPORT: OPTUNA TPE SAMPLER ({n_trials} TRIALS)\n")
        f.write("=" * 70 + "\n")
        f.write(f" Search Phase CV Acc  : {best_cv:.2f}%\n")
        f.write(f" Refit Test Accuracy  : {test_acc:.2f}%\n")
        f.write(f" Discovered Model     : {m_type}\n")
        f.write(f" Discovered FE        : {fe_strat}\n")
        f.write(f" Search Runtime       : {search_time/60:.1f} minutes\n")
        f.write("=" * 70 + "\n")
    return test_acc


def run_flaml_baseline(out_dir, time_budget=585):
    log_file = os.path.join(out_dir, "flaml.txt")
    print("\n" + "=" * 70)
    print(f" RUNNING BASELINE 4: FLAML AUTOML (TIME BUDGET: {time_budget}s / ~10m)")
    print("=" * 70)

    try:
        from flaml import AutoML
    except ImportError:
        print("  [!] FLAML not installed. Please run: pip install flaml")
        return None

    X_tr, X_te, y_tr, y_te = load_covertype_splits()
    automl = AutoML()
    settings = {
        "time_budget": time_budget,
        "metric": "accuracy",
        "task": "classification",
        "log_file_name": os.path.join(out_dir, "flaml_run.log"),
        "seed": 42,
        "estimator_list": ["lgbm", "rf", "extra_tree", "xgboost"]
    }

    start = time.time()
    automl.fit(X_train=X_tr, y_train=y_tr, **settings)
    total_time = time.time() - start

    preds = automl.predict(X_te)
    test_acc = float(np.mean(preds == y_te)) * 100.0

    print(f"  [RESULT] FLAML Test Accuracy : {test_acc:.2f}%")
    print(f"  [RESULT] FLAML Best Model    : {automl.best_estimator}")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f" BASELINE REPORT: FLAML AUTOML\n")
        f.write("=" * 70 + "\n")
        f.write(f" Time Budget Allotted : {time_budget} seconds\n")
        f.write(f" Best Estimator Found : {automl.best_estimator}\n")
        f.write(f" Test Accuracy        : {test_acc:.2f}%\n")
        f.write(f" Actual Search Time   : {total_time/60:.1f} minutes\n")
        f.write(f" Best Config          : {json.dumps(automl.best_config, indent=2)}\n")
        f.write("=" * 70 + "\n")
    return test_acc


def main():
    parser = argparse.ArgumentParser(description="Covertype Competitor Baselines Runner")
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "extratrees", "random", "optuna", "flaml"],
                        help="Which baseline to run (default: all)")
    parser.add_argument("--trials", type=int, default=120, help="Trials for HPO baselines")
    parser.add_argument("--time_budget", type=int, default=585, help="Time budget for FLAML in seconds")
    parser.add_argument("--run_name", type=str, default=None, help="Daily run folder name (e.g. Test1_19_Nov)")
    parser.add_argument("--output_dir", type=str, default=None, help="Direct path to output directory")
    args = parser.parse_args()

    if args.output_dir:
        out_dir = args.output_dir
    elif args.run_name:
        out_dir = os.path.join(_ROOT_DIR, "results_forest", args.run_name, "baselines")
    else:
        out_dir = os.path.join(_ROOT_DIR, "results_forest", "baselines")
    os.makedirs(out_dir, exist_ok=True)

    if args.test in ["all", "extratrees"]:
        run_default_extratrees(out_dir)
    if args.test in ["all", "random"]:
        run_random_search(out_dir, n_trials=args.trials)
    if args.test in ["all", "optuna"]:
        run_optuna_baseline(out_dir, n_trials=args.trials)
    if args.test in ["all", "flaml"]:
        run_flaml_baseline(out_dir, time_budget=args.time_budget)


if __name__ == "__main__":
    main()
