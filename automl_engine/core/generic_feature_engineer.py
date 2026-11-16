"""
Dataset-Agnostic Generic Feature Engineering Module
====================================================
Automatically discovers continuous and discrete feature patterns in any arbitrary
tabular dataset matrix (n_samples, n_features) without requiring domain knowledge
or hardcoded column semantics.

Transforms:
- 'none'        : Identity passthrough of raw input matrix.
- 'stats'       : Appends generic row-level statistics across continuous variables
                  (mean, standard deviation, min, max).
- 'interactions': Row statistics + pairwise differences and Euclidean norms across
                  top continuous features by variance.
"""
import numpy as np


class GenericFeatureEngineer:
    def __init__(self, strategy="none", max_interaction_features=6):
        self.strategy = strategy
        self.max_interaction_features = max_interaction_features
        self.continuous_idx = []
        self.top_cols = []

    def fit(self, X):
        if self.strategy == "none":
            return self

        # Automatically detect continuous features based on sample cardinality
        n_samples = min(2000, X.shape[0])
        sample_slice = X[:n_samples]
        
        self.continuous_idx = [
            i for i in range(X.shape[1])
            if len(np.unique(sample_slice[:, i])) > 15
        ]

        if not self.continuous_idx:
            self.continuous_idx = list(range(X.shape[1]))

        if self.strategy == "interactions" and len(self.continuous_idx) > 1:
            # Rank continuous features by variance to avoid combinatorial explosion
            variances = np.var(X[:, self.continuous_idx], axis=0)
            sorted_order = np.argsort(variances)[::-1]
            self.top_cols = [self.continuous_idx[i] for i in sorted_order[:self.max_interaction_features]]

        return self

    def transform(self, X):
        if self.strategy == "none" or not self.continuous_idx:
            return X

        X_num = X[:, self.continuous_idx]
        
        # Row summary statistics
        row_mean = np.mean(X_num, axis=1, keepdims=True)
        row_std = np.std(X_num, axis=1, keepdims=True)
        row_min = np.min(X_num, axis=1, keepdims=True)
        row_max = np.max(X_num, axis=1, keepdims=True)
        new_blocks = [X, row_mean, row_std, row_min, row_max]

        # Pairwise interaction features
        if self.strategy == "interactions" and len(self.top_cols) > 1:
            pairs = []
            for i in range(len(self.top_cols)):
                for j in range(i + 1, len(self.top_cols)):
                    ci = X[:, self.top_cols[i]]
                    cj = X[:, self.top_cols[j]]
                    pairs.append((ci - cj)[:, None])
                    pairs.append(np.sqrt(ci**2 + cj**2)[:, None])
            new_blocks.extend(pairs)

        return np.hstack(new_blocks).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

