"""
EvoSynth Scaled Covertype Benchmark Runner
===========================================
Runs EvoSynth at scaled budgets (e.g., 300, 500) on the UCI Forest Covertype dataset.
Includes atomic checkpointing every 10 evaluations, full log capture, and post-search
generalization refit on 60,000 train / 20,000 held-out test samples.

Outputs:
  - results_forest/evosynth/run_<budget>.txt
  - results_forest/evosynth/checkpoints/checkpoint_<budget>.json
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


from automl_engine.core.search_space import get_xgboost_search_space
from automl_engine.core.covtype_evaluator import CovertypeEvaluator
from automl_engine.engine.controller import DynamicOptimizer
from ConfigSpace import Configuration
import ConfigSpace.hyperparameters as CSH


class TeeLogger:
    """Mirrors console output live to a log file."""
    def __init__(self, log_path, stream):
        self.stream = stream
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.stream.write(message)
        self.stream.flush()
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def save_atomic_checkpoint(checkpoint_path, data):
    tmp_path = checkpoint_path + ".tmp"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    if os.path.exists(checkpoint_path):
        os.replace(tmp_path, checkpoint_path)
    else:
        os.rename(tmp_path, checkpoint_path)


def decode_vector_to_dict(vector, space):
    valid_vec = np.copy(vector)
    for i, hp_name in enumerate(list(space.keys())):
        hp = space.get_hyperparameter(hp_name)
        if isinstance(hp, CSH.CategoricalHyperparameter):
            num_choices = len(hp.choices)
            idx = int(np.floor(valid_vec[i] * num_choices))
            idx = min(idx, num_choices - 1)
            valid_vec[i] = float(idx)
    cfg = Configuration(space, vector=valid_vec)
    return {k: cfg[k] for k in list(space.keys())}


def run_benchmark(budget=300, pop_size=30, subsample=20000, checkpoint_interval=10, run_name=None, output_dir=None):
    if output_dir:
        target_dir = output_dir
    elif run_name:
        target_dir = os.path.join(_ROOT_DIR, "results_forest", run_name, "evosynth")
    else:
        target_dir = os.path.join(_ROOT_DIR, "results_forest", "evosynth")

    ckpt_dir = os.path.join(target_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    log_path = os.path.join(target_dir, f"run_{budget}.txt")
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{budget}.json")

    tee = TeeLogger(log_path, sys.stdout)
    sys.stdout = tee

    start_time = time.time()
    try:
        print("=" * 80)
        print(f" EVOSYNTH SCALED EXPERIMENT — UCI FOREST COVERTYPE (BUDGET: {budget})")
        print(f" Population: {pop_size} | Checkpoint Interval: Every {checkpoint_interval} evals")
        print(f" Dataset Subsample: {subsample:,} (2-Fold Stratified CV) | Latent Dims: 9")
        print("=" * 80)

        space = get_xgboost_search_space(include_model_type=True, include_fe_strategy=True)
        evaluator = CovertypeEvaluator(subsample_size=subsample)

        # Wrap evaluator to trigger atomic checkpoints every N evals
        original_evaluate = evaluator.evaluate
        eval_count = 0
        best_so_far = 0.0
        best_cfg = None

        def monitored_evaluate(individual):
            nonlocal eval_count, best_so_far, best_cfg
            score = original_evaluate(individual)
            eval_count += 1
            if score > best_so_far:
                best_so_far = score
                best_cfg = decode_vector_to_dict(individual.vector, space)

            if eval_count % checkpoint_interval == 0 or eval_count == budget:
                ckpt_data = {
                    "budget": budget,
                    "evals_used": eval_count,
                    "best_cv_accuracy": float(best_so_far),
                    "best_config": best_cfg,
                    "elapsed_seconds": round(time.time() - start_time, 2),
                    "status": "in_progress" if eval_count < budget else "completed",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                save_atomic_checkpoint(ckpt_path, ckpt_data)
            return score

        evaluator.evaluate = monitored_evaluate

        # Determine tier iters based on budget
        t1_iters = max(3, int(budget * 0.03))
        k_iters = max(2, int(budget * 0.02))

        optimizer = DynamicOptimizer(
            search_space=space,
            evaluator=evaluator,
            budget=budget,
            pop_size=pop_size,
            t1_iters=t1_iters,
            k_iters=k_iters
        )

        best = optimizer.run()

        # Step 6: Full Generalization Refit on 60k / 20k unseen test data
        print("\n" + "=" * 80)
        print(" STEP 6: POST-SEARCH GENERALIZATION REFIT ON HELD-OUT TEST DATA")
        print("=" * 80)
        print("  Evaluating winning architecture & hyperparameters at full capacity...")

        from sklearn.datasets import fetch_covtype
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        import xgboost as xgb
        import lightgbm as lgb
        from automl_engine.core.generic_feature_engineer import GenericFeatureEngineer

        _data_home = os.path.join(_ROOT_DIR, ".data")
        data = fetch_covtype(data_home=_data_home)
        X_full = data.data.astype(np.float32)
        y_full = data.target.astype(int) - 1

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_full, y_full, train_size=60000, test_size=20000, stratify=y_full, random_state=42
        )

        winning_cfg = decode_vector_to_dict(best.vector, space)
        m_type = winning_cfg.get("model_type", "ExtraTrees")
        fe_strat = winning_cfg.get("fe_strategy", "none")
        lr = winning_cfg.get("learning_rate", 0.1)
        n_est = winning_cfg.get("n_estimators", 150)
        depth = winning_cfg.get("max_depth", 8)
        sub = winning_cfg.get("subsample", 0.8)
        colsample = winning_cfg.get("colsample_bytree", 0.8)
        reg_alpha = winning_cfg.get("reg_alpha", 0.0)

        fe = GenericFeatureEngineer(strategy=fe_strat)
        if fe_strat != "none":
            print(f"  [FE] Applying winning feature engineering strategy: '{fe_strat}'")
            X_tr = fe.fit_transform(X_tr)
            X_te = fe.transform(X_te)

        print(f"  [MODEL] Training winning architecture: '{m_type}' on 60,000 samples...")
        if m_type == "ExtraTrees":
            model = ExtraTreesClassifier(
                n_estimators=max(n_est, 150),
                max_depth=depth * 3 if depth < 10 else None,
                max_features=colsample,
                random_state=42,
                n_jobs=-1
            )
        elif m_type == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=max(n_est, 150),
                max_depth=depth * 3 if depth < 10 else None,
                max_features=colsample,
                random_state=42,
                n_jobs=-1
            )
        elif m_type == "LightGBM":
            model = lgb.LGBMClassifier(
                learning_rate=lr,
                n_estimators=max(n_est, 250),
                max_depth=depth,
                subsample=sub,
                colsample_bytree=colsample,
                reg_alpha=reg_alpha,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        else:
            model = xgb.XGBClassifier(
                learning_rate=lr,
                n_estimators=max(n_est, 200),
                max_depth=depth,
                subsample=sub,
                colsample_bytree=colsample,
                reg_alpha=reg_alpha,
                random_state=42,
                n_jobs=-1,
                eval_metric="mlogloss"
            )

        model.fit(X_tr, y_tr)
        test_preds = model.predict(X_te)
        test_acc = float(np.mean(test_preds == y_te)) * 100.0

        elapsed = time.time() - start_time
        print("\n  +----------------------------------------------------------------+")
        print(f"  | EVO-SYNTH FINAL BENCHMARK SUMMARY (BUDGET: {budget})          |")
        print("  +----------------------------------------------------------------+")
        print(f"  | Total Search Time              : {elapsed/60:.1f} minutes")
        print(f"  | Search Phase CV Accuracy       : {best.fitness:.2f}% (20k slice)")
        print(f"  | Generalization Test Accuracy   : {test_acc:.2f}% (60k train, 20k test)")
        print(f"  | Optimal Architecture           : {m_type}")
        print(f"  | Optimal Feature Engineering    : {fe_strat}")
        print("  +----------------------------------------------------------------+")

        final_ckpt = {
            "budget": budget,
            "evals_used": budget,
            "best_cv_accuracy": float(best.fitness),
            "generalization_test_accuracy": float(test_acc),
            "optimal_architecture": m_type,
            "optimal_fe_strategy": fe_strat,
            "best_config": winning_cfg,
            "elapsed_seconds": round(elapsed, 2),
            "status": "completed",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        save_atomic_checkpoint(ckpt_path, final_ckpt)

    finally:
        sys.stdout = tee.stream
        tee.close()


def main():
    parser = argparse.ArgumentParser(description="EvoSynth Scaled Covertype Benchmark")
    parser.add_argument("--budget", type=int, default=300, choices=[120, 300, 500, 1000],
                        help="Evaluation budget (default: 300)")
    parser.add_argument("--pop_size", type=int, default=30, help="Population size (default: 30)")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                        help="Save checkpoint every N evals (default: 10)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Daily run folder name (e.g. Test1_19_Nov)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Direct path to output directory")
    args = parser.parse_args()

    run_benchmark(
        budget=args.budget,
        pop_size=args.pop_size,
        checkpoint_interval=args.checkpoint_interval,
        run_name=args.run_name,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
