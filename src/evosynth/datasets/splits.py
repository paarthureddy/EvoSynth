"""
Stratified splits shared by the search evaluator, the refit and every baseline,
so all methods see exactly the same rows.
"""
from sklearn.model_selection import train_test_split


def search_subsample(X, y, n_rows, random_state):
    """Stratified n_rows sample used during search (the rest is discarded)."""
    X_s, _, y_s, _ = train_test_split(X, y, train_size=n_rows, stratify=y, random_state=random_state)
    return X_s, y_s


def refit_split(X, y, train_rows, test_rows, random_state):
    """Stratified train/test split for the post-search refit (test rows never seen in search)."""
    return train_test_split(X, y, train_size=train_rows, test_size=test_rows,
                            stratify=y, random_state=random_state)
