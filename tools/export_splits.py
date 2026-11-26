"""
Export the exact data splits EvoSynth uses (run.data_seed) as row-index files, for the
EvoSynth-Bench dataset release (Zenodo 10.5281/zenodo.23645119).

    py -3.14 tools/export_splits.py OUT_DIR

Writes OUT_DIR/splits/<dataset>.csv (row_index, split) and OUT_DIR/datasets.csv (catalogue),
and checks that the indices reproduce datasets.partition exactly. Row indices refer to the row
order of the source as loaded by src/evosynth/datasets/loaders.py (sklearn fetch_covtype /
OpenML / torchvision order).
"""
import csv
import sys
from pathlib import Path

import numpy as np

from evosynth import datasets
from evosynth.config import get_settings
from evosynth.datasets.splits import partition, sample

SOURCES = {   # where each dataset comes from and its licence (for the catalogue)
    "covertype": ("https://archive.ics.uci.edu/dataset/31/covertype", "CC BY 4.0"),
    "higgs": ("https://www.openml.org/d/23512", "Public (OpenML); UCI original CC BY 4.0"),
    "adult": ("https://www.openml.org/d/1590", "Public (OpenML); UCI original CC BY 4.0"),
    "airlines": ("https://www.openml.org/d/1169", "Public (OpenML)"),
    "poker_hand": ("https://www.openml.org/d/1569", "Public (OpenML); UCI original CC BY 4.0"),
    "miniboone": ("https://www.openml.org/d/41150", "CC0 (OpenML); UCI original CC BY 4.0"),
    "jannis": ("https://www.openml.org/d/41168", "Public (OpenML)"),
    "fashion_mnist": ("https://github.com/zalandoresearch/fashion-mnist", "MIT"),
}


def tabular(name, spec, seed):
    X, y = datasets.load(name)
    idx = np.arange(len(y)).reshape(-1, 1)
    parts = partition(idx, y, spec.test_rows, spec.search_subsample, spec.refit_train_rows, seed)
    real = datasets.partition(name)
    for k in ("test", "search", "train"):            # indices reproduce the real split exactly
        assert np.array_equal(X[parts[k][0][:, 0]], real[k][0]), (name, k)
        assert np.array_equal(y[parts[k][0][:, 0]], real[k][1]), (name, k)
    test, search, train = (set(parts[k][0][:, 0].tolist()) for k in ("test", "search", "train"))
    assert not test & search and not test & train
    rows = [(i, "test") for i in sorted(test)]
    rows += [(i, "search+train" if i in train else "search") for i in sorted(search)]
    rows += [(i, "train") for i in sorted(train - search)]
    info = dict(rows=len(y), features=X.shape[1], classes=len(np.unique(y)), test=len(test),
                search=len(search), train=len(train))
    return rows, info


def image(name, spec, seed):
    from sklearn.model_selection import train_test_split
    data = datasets.load(name)
    X, y = data["train"]
    idx = np.arange(len(y)).reshape(-1, 1)
    i_s, y_s = sample(idx, y, spec.search_train_rows + spec.search_val_rows, seed)
    i_tr, i_va, _, _ = train_test_split(i_s, y_s, test_size=spec.search_val_rows, random_state=seed, stratify=y_s)
    rows = sorted([(int(i), "search_train") for i in i_tr[:, 0]] + [(int(i), "search_val") for i in i_va[:, 0]])
    info = dict(rows=len(y) + len(data["test"][1]), features=int(np.prod(X.shape[1:])),
                classes=len(np.unique(y)), test=len(data["test"][1]), search=len(i_s), train=len(i_tr))
    return rows, info


def main(out):
    s = get_settings()
    seed = s.run.data_seed
    (out / "splits").mkdir(parents=True, exist_ok=True)
    catalogue = []
    for name in datasets.available():
        spec = s.datasets[name]
        rows, info = (tabular if spec.kind == "tabular" else image)(name, spec, seed)
        with open(out / "splits" / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["row_index", "split"])
            w.writerows(rows)
        url, licence = SOURCES[name]
        catalogue.append({"dataset": name, "kind": spec.kind, "source": spec.source,
                          "openml_id": spec.get("openml_id", ""), "url": url, "source_licence": licence,
                          **info, "metric": spec.metric, "data_seed": seed})
        print(f"{name:<14} {info}")
    with open(out / "datasets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(catalogue[0]))
        w.writeheader()
        w.writerows(catalogue)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
