"""
One loader per data source. Each takes (name, spec) and returns (X, y):
  tabular: X float32 (n, d) with no missing values, y int64 labels 0..K-1
  image:   {"train": (X, y), "test": (X, y)} with X float32 (n, 1, H, W) in [0, 1]

Raw downloads and the processed arrays are cached under evosynth.paths.DATA_DIR (.data/),
so each dataset is downloaded and converted once. Processed tabular data is stored in
.data/<name>/<name>.npz; delete that file to rebuild it from the raw download.

Preprocessing (identical for every method, done once on the whole dataset; no label is used):
  - labels re-coded to 0..K-1 (sorted original labels)
  - categorical / text columns → integer codes (categories sorted), missing → its own code
  - missing numeric values → the column median
"""
import numpy as np

from evosynth.paths import DATA_DIR


def _encode_labels(y):
    classes, codes = np.unique(np.asarray(y), return_inverse=True)
    return codes.astype(np.int64), classes


def _cached(name, build):
    """Return the processed (X, y) from .data/<name>/<name>.npz, building it on the first call."""
    folder = DATA_DIR / name
    path = folder / f"{name}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as d:
            return d["X"], d["y"]
    X, y = build(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"{name}.tmp.npz"
    np.savez_compressed(tmp, X=X, y=y)
    tmp.replace(path)
    return X, y


def load_covertype(name, spec):
    """UCI Forest Covertype (581,012 x 54, 7 classes) via sklearn.datasets.fetch_covtype."""
    from sklearn.datasets import fetch_covtype
    data = fetch_covtype(data_home=str(DATA_DIR))
    y, _ = _encode_labels(data.target)
    return data.data.astype(np.float32), y


def _frame_to_arrays(frame, target):
    """pandas DataFrame from OpenML → (X float32, y int64) with the preprocessing above."""
    import pandas as pd

    y, _ = _encode_labels(frame[target].astype(str).to_numpy(dtype=object))
    cols = []
    for col in frame.columns:
        if col == target:
            continue
        s = frame[col]
        if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or pd.api.types.is_string_dtype(s):
            cats = sorted(s.dropna().astype(str).unique())
            lookup = {c: i for i, c in enumerate(cats)}
            codes = s.astype(object).map(lambda v: lookup.get(str(v), len(cats)) if pd.notna(v) else len(cats))
            cols.append(codes.to_numpy(dtype=np.float32))
        else:
            v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
            if np.isnan(v).any():
                v = np.where(np.isnan(v), np.nanmedian(v), v)
            cols.append(v.astype(np.float32))
    return np.column_stack(cols), y


def load_openml(name, spec):
    """Any OpenML dataset by spec.openml_id (download cached in .data/openml/)."""
    def build(_folder):
        from sklearn.datasets import fetch_openml
        print(f"  [DATASET] Downloading OpenML {spec['openml_id']} ({name}) into {DATA_DIR / 'openml'} ...")
        bunch = fetch_openml(data_id=int(spec["openml_id"]), data_home=str(DATA_DIR / "openml"),
                             as_frame=True, parser="auto")
        target = bunch.target.name if hasattr(bunch.target, "name") else bunch.target_names[0]
        frame = bunch.frame if bunch.frame is not None else bunch.data.assign(**{target: bunch.target})
        return _frame_to_arrays(frame, target)
    return _cached(name, build)


def load_fashion_mnist(name, spec):
    """Fashion-MNIST via torchvision (download cached in .data/fashion_mnist/)."""
    from torchvision.datasets import FashionMNIST

    root = DATA_DIR / name
    out = {}
    for split, train in (("train", True), ("test", False)):
        ds = FashionMNIST(root=str(root), train=train, download=True)
        X = ds.data.numpy().astype(np.float32)[:, None, :, :] / 255.0
        out[split] = (X, ds.targets.numpy().astype(np.int64))
    return out


# source (settings: datasets.<name>.source) → loader function
LOADERS = {
    "sklearn.fetch_covtype": load_covertype,
    "openml": load_openml,
    "torchvision.FashionMNIST": load_fashion_mnist,
}
