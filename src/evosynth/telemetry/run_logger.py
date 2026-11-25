"""
Run logging: mirror stdout to the run's log file (+ live 'log' events) and write one CSV
row per evaluated individual.

Files (inside the run folder, see evosynth.telemetry.run_context): run.log, population.csv.
"""
import sys

from evosynth.core.decoding import decode_vector_to_dict


def make_safe(stream):
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
    """stdout replacement: echo to console, write to the log file, emit 'log' events."""

    def __init__(self, original_stdout, log_path, reporter=None, mode="w"):
        self.original_stdout = make_safe(original_stdout)
        self.reporter = reporter
        self.log_file = open(log_path, mode, encoding="utf-8")

    def write(self, message):
        self.original_stdout.write(message)
        self.original_stdout.flush()
        # Flush immediately so the log survives crashes / Ctrl-C
        self.log_file.write(message)
        self.log_file.flush()
        if self.reporter is not None and message and message != "\n":
            self.reporter.emit("log", {"text": message})

    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def install_stdout_mirror(log_path, reporter=None, mode="w"):
    """Replace sys.stdout with a LoggerWriter writing to log_path; returns the writer
    (restore with `sys.stdout = writer.original_stdout`)."""
    sys.stdout = LoggerWriter(sys.stdout, log_path, reporter, mode)
    return sys.stdout


def _phase(algo_name):
    if algo_name == "Init":
        return "Init"
    if algo_name.startswith("T1"):
        return "T1"
    if algo_name.startswith("T2"):
        return "T2"
    return None


class PopulationLogger:
    """One CSV row per evaluation: Eval_ID, Phase (Init/T1/T2), Algorithm, Fitness, <decoded hyperparameters>."""

    def __init__(self, space, csv_path):
        self.space = space
        self.file = open(csv_path, "w", encoding="utf-8")
        self.file.write("Eval_ID,Phase,Algorithm,Fitness," + ",".join(space.keys()) + "\n")
        self.file.flush()

    def log(self, eval_id, algo_name, ind):
        phase = _phase(algo_name)
        if phase is None:
            return
        params = decode_vector_to_dict(ind.vector, self.space)
        self.file.write(f"{eval_id},{phase},{algo_name},{ind.fitness:.4f},"
                        + ",".join(str(params[k]) for k in self.space.keys()) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()
