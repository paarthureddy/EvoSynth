"""
Crash-safe experiment state: atomic JSON/pickle writes (write to .tmp, then rename,
so a checkpoint is never half-written). Console logging lives in run_context (run.log).
"""
import json
import os
import pickle


def _atomic_replace(tmp_path, final_path):
    if os.path.exists(final_path):
        os.replace(tmp_path, final_path)
    else:
        os.rename(tmp_path, final_path)


def _json_default(obj):
    """numpy scalars/arrays -> Python values; anything else -> str."""
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def atomic_json_dump(file_path, data):
    """Write `data` as indented JSON to file_path atomically."""
    tmp_path = file_path + ".tmp"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)
    _atomic_replace(tmp_path, file_path)


def atomic_pickle_dump(file_path, data):
    """Pickle `data` to file_path atomically."""
    tmp_path = file_path + ".tmp"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f)
    _atomic_replace(tmp_path, file_path)
