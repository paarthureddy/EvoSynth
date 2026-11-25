"""
Run logging: mirror stdout to a log file (+ live 'log' events) and write one CSV row
per evaluated individual.

Files (under evosynth.paths.RESULTS_DIR): run_output.txt, csv/warmup_population.csv,
csv/tier1_population.csv, csv/tier2_population.csv.
"""
import os
import sys

from evosynth.core.decoding import decode_vector_to_dict


def _make_safe(stream):
    """Never crash on characters the console/pipe encoding cannot represent."""
    try:
        if getattr(stream, "isatty", lambda: False)():
            stream.reconfigure(errors="replace")
        else:
            stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    return stream


class LoggerWriter:
    """stdout replacement: echo to console, append to run_output.txt, emit 'log' events."""

    def __init__(self, original_stdout, log_path, reporter):
        self.original_stdout = _make_safe(original_stdout)
        self.reporter = reporter
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.original_stdout.write(message)
        self.original_stdout.flush()
        # Flush immediately so the log survives crashes / Ctrl-C
        self.log_file.write(message)
        self.log_file.flush()
        if message and message != "\n":
            self.reporter.emit("log", {"text": message})

    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()


def install_stdout_mirror(results_dir, reporter):
    """Replace sys.stdout with a LoggerWriter writing to <results_dir>/run_output.txt."""
    os.makedirs(results_dir, exist_ok=True)
    sys.stdout = LoggerWriter(sys.stdout, os.path.join(results_dir, "run_output.txt"), reporter)
    return sys.stdout


class PopulationLogger:
    """One CSV row per evaluation: Eval_ID, Algorithm, Fitness, <decoded hyperparameters>."""

    def __init__(self, space, csv_dir):
        self.space = space
        os.makedirs(csv_dir, exist_ok=True)
        self.files = {
            "Init": open(os.path.join(csv_dir, "warmup_population.csv"), "w", encoding="utf-8"),
            "T1": open(os.path.join(csv_dir, "tier1_population.csv"), "w", encoding="utf-8"),
            "T2": open(os.path.join(csv_dir, "tier2_population.csv"), "w", encoding="utf-8"),
        }
        header = "Eval_ID,Algorithm,Fitness," + ",".join(space.keys()) + "\n"
        for f in self.files.values():
            f.write(header)
            f.flush()

    def log(self, eval_id, algo_name, ind):
        if algo_name == "Init":
            f = self.files["Init"]
        elif algo_name.startswith("T1"):
            f = self.files["T1"]
        elif algo_name.startswith("T2"):
            f = self.files["T2"]
        else:
            return
        params = decode_vector_to_dict(ind.vector, self.space)
        f.write(f"{eval_id},{algo_name},{ind.fitness:.4f}," + ",".join(str(params[k]) for k in self.space.keys()) + "\n")
        f.flush()
