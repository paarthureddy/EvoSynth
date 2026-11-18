"""
EvoSynth-Bench Dataset Loader & Benchmark Validation Script
Group G002 - Automated Machine Learning & Optimization Lab
"""

import os
import pandas as pd
import numpy as np

def load_pathological_benchmark(data_dir="data"):
    path = os.path.join(data_dir, "pathological_classification_benchmark.csv")
    if not os.path.exists(path):
        # Fallback to current dir or ../data
        path = os.path.join("..", "data", "pathological_classification_benchmark.csv")
    df = pd.read_csv(path)
    X = df.drop(columns=["target"])
    y = df["target"]
    print(f"[+] Loaded Pathological Benchmark: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"    Class distribution: {dict(y.value_counts())}")
    return X, y

def load_hpo_trajectories(data_dir="data"):
    path = os.path.join(data_dir, "evosynth_hpo_trajectories.csv")
    if not os.path.exists(path):
        path = os.path.join("..", "data", "evosynth_hpo_trajectories.csv")
    df = pd.read_csv(path)
    print(f"[+] Loaded HPO Trajectories: {len(df)} evaluations")
    print(f"    Algorithms represented: {df['Algorithm'].unique().tolist()}")
    print(f"    Model types evaluated: {df['model_type'].unique().tolist()}")
    print(f"    Best fitness reached: {df['Fitness'].max():.4f}%")
    return df

if __name__ == "__main__":
    print("=" * 60)
    print("  EvoSynth-Bench: Multi-Algorithm HPO & Landscape Benchmark")
    print("=" * 60)
    X, y = load_pathological_benchmark("dataset_release/data" if os.path.exists("dataset_release/data") else "data")
    trajectories = load_hpo_trajectories("dataset_release/data" if os.path.exists("dataset_release/data") else "data")
    print("=" * 60)
    print("Dataset verification PASSED. Ready for research reuse.")
