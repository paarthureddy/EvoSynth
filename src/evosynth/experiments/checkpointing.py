"""
Crash-safe experiment state: atomic JSON/pickle writes (write to .tmp, then rename,
so a checkpoint is never half-written) and a stdout tee that mirrors a run to a log file.
"""
import json
import os
import pickle


def _atomic_replace(tmp_path, final_path):
    if os.path.exists(final_path):
        os.replace(tmp_path, final_path)
    else:
        os.rename(tmp_path, final_path)


def atomic_json_dump(file_path, data):
    """Write `data` as indented JSON to file_path atomically."""
    tmp_path = file_path + ".tmp"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    _atomic_replace(tmp_path, file_path)


def atomic_pickle_dump(file_path, data):
    """Pickle `data` to file_path atomically."""
    tmp_path = file_path + ".tmp"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f)
    _atomic_replace(tmp_path, file_path)


class TeeLogger:
    """Mirrors console output live to a log file (UTF-8)."""

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
