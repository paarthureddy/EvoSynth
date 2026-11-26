"""
The score every method is judged by: one function, chosen per dataset by
[datasets.<name>].metric (settings/datasets.toml). Returns a fraction in [0, 1];
callers report it as a percentage.

  accuracy           share of correct predictions
  balanced_accuracy  mean recall over classes (robust to imbalance)
  roc_auc            area under the ROC curve (binary datasets; uses predicted probabilities)
"""
import numpy as np

METRICS = ("accuracy", "balanced_accuracy", "roc_auc")


def score(model, X, y, metric):
    """Score a fitted classifier on (X, y)."""
    if metric == "accuracy":
        return float(np.mean(model.predict(X) == y))
    if metric == "balanced_accuracy":
        from sklearn.metrics import balanced_accuracy_score
        return float(balanced_accuracy_score(y, model.predict(X)))
    if metric == "roc_auc":
        from sklearn.metrics import roc_auc_score
        classes = list(model.classes_)
        if len(classes) != 2 or len(np.unique(y)) != 2:
            raise ValueError("roc_auc needs a binary task with both classes present")
        return float(roc_auc_score(y == classes[1], model.predict_proba(X)[:, 1]))
    raise ValueError(f"unknown metric '{metric}'. Choose from {METRICS}")
