"""
EvoSynth Real-World Benchmark: UCI Forest Covertype Dataset
============================================================
Runs the full Dynamic Multi-Armed Bandit Meta-Optimizer on the Forest Covertype
dataset (581,012 samples, 54 features, 7-class multiclass classification).

This is a REAL published UCI benchmark — not synthetic data.

Usage:
    python run_covertype.py              # Fast demo (120 evals, ~3-5 min)
    python run_covertype.py --full       # Full benchmark (500 evals, ~15-20 min)
"""
import argparse
from automl_engine.core.search_space import get_xgboost_search_space
from automl_engine.core.covtype_evaluator import CovertypeEvaluator
from automl_engine.engine.controller import DynamicOptimizer


def main():
    parser = argparse.ArgumentParser(description="EvoSynth Covertype Real-World Benchmark")
    parser.add_argument("--budget", type=int, default=120, help="Total evaluation budget")
    parser.add_argument("--pop_size", type=int, default=20, help="Population size")
    parser.add_argument("--t1_iters", type=int, default=3, help="Tier 1 competition iterations")
    parser.add_argument("--k_iters", type=int, default=2, help="Checkpoint interval")
    parser.add_argument("--subsample", type=int, default=20000, help="Stratified subsample size")
    parser.add_argument("--full", action="store_true", help="Run full benchmark (500 evals, pop=40)")
    args = parser.parse_args()

    if args.full:
        budget = 500
        pop_size = 40
        t1_iters = 5
        k_iters = 3
    else:
        budget = args.budget
        pop_size = args.pop_size
        t1_iters = args.t1_iters
        k_iters = args.k_iters

    print("=" * 70)
    print(" EVOSYNTH: REAL-WORLD BENCHMARK — UCI FOREST COVERTYPE DATASET")
    print(f" Mode: {'Full Benchmark' if args.full else 'Fast Demo'} | Budget: {budget} | Pop: {pop_size}")
    print(f" Dataset: 581,012 samples → {args.subsample:,} stratified subsample | 54 features | 7 classes")
    print("=" * 70)

    # 1. Search space with model switching (XGBoost / LightGBM / RandomForest / ExtraTrees) & generic FE
    space = get_xgboost_search_space(include_model_type=True, include_fe_strategy=True)

    # 2. Real-world evaluator on Forest Covertype
    evaluator = CovertypeEvaluator(subsample_size=args.subsample)

    # 3. Dynamic Multi-Armed Bandit Meta-Optimizer
    optimizer = DynamicOptimizer(
        search_space=space,
        evaluator=evaluator,
        budget=budget,
        pop_size=pop_size,
        t1_iters=t1_iters,
        k_iters=k_iters
    )

    best = optimizer.run()

    # 4. Generalization Evaluation on Expanded Unseen Held-out Split
    print("\n" + "=" * 80)
    print(" STEP 6: POST-SEARCH GENERALIZATION REFIT ON HELD-OUT TEST DATA")
    print("=" * 80)
    print("  Evaluating winning architecture & hyperparameters at full capacity...")
    try:
        from sklearn.datasets import fetch_covtype
        from sklearn.model_selection import train_test_split
        from ConfigSpace import Configuration
        import ConfigSpace.hyperparameters as CSH
        import os
        import numpy as np

        _data_home = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data")
        data = fetch_covtype(data_home=_data_home)
        X_full = data.data.astype(np.float32)
        y_full = data.target.astype(int) - 1

        # Train on 60k, evaluate on 20k unseen test samples
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_full, y_full, train_size=60000, test_size=20000, stratify=y_full, random_state=42
        )

        valid_vec = np.copy(best.vector)
        for i, hp_name in enumerate(list(space.keys())):
            hp = space.get_hyperparameter(hp_name)
            if isinstance(hp, CSH.CategoricalHyperparameter):
                num_choices = len(hp.choices)
                idx = int(np.floor(valid_vec[i] * num_choices))
                idx = min(idx, num_choices - 1)
                valid_vec[i] = float(idx)

        cfg = Configuration(space, vector=valid_vec)
        m_type = cfg.get("model_type", "XGBoost")
        fe_strat = cfg.get("fe_strategy", "none")
        lr = cfg.get("learning_rate", 0.1)
        n_est = cfg.get("n_estimators", 150)
        depth = cfg.get("max_depth", 8)
        sub = cfg.get("subsample", 0.8)
        colsample = cfg.get("colsample_bytree", 0.8)
        reg_alpha = cfg.get("reg_alpha", 0.0)

        from automl_engine.core.generic_feature_engineer import GenericFeatureEngineer
        fe = GenericFeatureEngineer(strategy=fe_strat)
        if fe_strat != "none":
            print(f"  [FE] Applying winning feature engineering strategy: '{fe_strat}'")
            X_tr = fe.fit_transform(X_tr)
            X_te = fe.transform(X_te)
        print(f"  [MODEL] Training winning architecture: '{m_type}' on 60,000 samples...")

        if m_type == "ExtraTrees":
            from sklearn.ensemble import ExtraTreesClassifier
            model = ExtraTreesClassifier(
                n_estimators=max(n_est, 150),
                max_depth=depth * 3 if depth < 10 else None,
                max_features=colsample,
                random_state=42,
                n_jobs=-1
            )
        elif m_type == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=max(n_est, 150),
                max_depth=depth * 3 if depth < 10 else None,
                max_features=colsample,
                random_state=42,
                n_jobs=-1
            )
        elif m_type == "LightGBM":
            import lightgbm as lgb
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
            import xgboost as xgb
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

        print(f"\n  +----------------------------------------------------------------+")
        print(f"  | EVO-SYNTH FINAL BENCHMARK SUMMARY                             |")
        print(f"  +----------------------------------------------------------------+")
        print(f"  | Search Phase CV Accuracy       : {best.fitness:.2f}% (Search slice)")
        print(f"  | Generalization Test Accuracy   : {test_acc:.2f}% (60k train, 20k test)")
        print(f"  | Optimal Architecture           : {m_type}")
        print(f"  | Optimal Feature Engineering    : {fe_strat}")
        print(f"  +----------------------------------------------------------------+")
    except Exception as e:
        print(f"  Refit skipped: {e}")

    return best


if __name__ == "__main__":
    main()

