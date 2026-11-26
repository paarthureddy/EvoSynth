"""
One folder per run:  results/<experiment>/<dataset>/<variant>/seed_<k>/

    manifest.json    what produced the run: command, resolved settings, git commit, library
                     versions, start/end time, status (running / completed / failed / interrupted)
    result.json      the run's final numbers (written by the experiment)
    run.log          everything printed to the console (UTF-8)
    events.jsonl     engine events (step, progress, tier_state, metric, composite, switch)
    population.csv   one row per evaluation of the EvoSynth run, when there is one
    checkpoint.json  / checkpoints/   resume state, where the experiment has it

Only manifest.json, result.json and summary/ folders are committed to git (see .gitignore).
The seed in the folder name is [run].seed; RunContext also seeds the global generators with it.
"""
import datetime as _dt
import importlib.metadata as _md
import os
import platform
import shlex
import subprocess
import sys
import time

from evosynth.config import get_settings
from evosynth.experiments.checkpointing import atomic_json_dump
from evosynth.paths import PROJECT_ROOT, RESULTS_DIR
from evosynth.seeding import seed_everything
from evosynth.telemetry.reporter import EventReporter, set_reporter
from evosynth.telemetry.run_logger import install_stdout_mirror

# Libraries recorded in manifest.json (None when not installed)
TRACKED_PACKAGES = ("evosynth", "numpy", "scipy", "pandas", "scikit-learn", "ConfigSpace", "xgboost",
                    "lightgbm", "cma", "torch", "torchvision", "optuna", "flaml")


def _slug(text):
    return str(text).strip().lower().replace("-", "_").replace(" ", "_")


def run_path(experiment, dataset, variant, seed=None, results_dir=None):
    """Folder of one run (not created)."""
    seed = get_settings().run.seed if seed is None else seed
    root = RESULTS_DIR if results_dir is None else results_dir
    return os.path.join(str(root), _slug(experiment), _slug(dataset), _slug(variant), f"seed_{seed}")


def _now():
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _git(*args):
    try:
        out = subprocess.run(["git", "-C", str(PROJECT_ROOT), *args], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_info():
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status)}


def environment_info():
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = _md.version(name)
        except _md.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


class RunContext:
    """
    with RunContext("scaling", "covertype", "budget_500") as run:
        ...                                   # console output -> run.log, events -> events.jsonl
        run.write_result({...})               # -> result.json

    resume=True keeps an existing run.log / events.jsonl (appends) for experiments that
    resume from their own checkpoints (ablation, Optuna).
    """

    def __init__(self, experiment, dataset, variant, seed=None, results_dir=None, resume=False):
        self.experiment, self.dataset, self.variant = experiment, dataset, _slug(variant)
        self.seed = get_settings().run.seed if seed is None else seed
        self.path = run_path(experiment, dataset, variant, self.seed, results_dir)
        self.resume = resume
        self._manifest = None
        self._t0 = None

    def file(self, name):
        """Absolute path of a file inside the run folder."""
        return os.path.join(self.path, name)

    # -- lifecycle --------------------------------------------------------------
    def __enter__(self):
        os.makedirs(self.path, exist_ok=True)
        seed_everything(self.seed)
        for stale in ("result.json",) + (() if self.resume else ("events.jsonl",)):
            if os.path.exists(self.file(stale)):
                os.remove(self.file(stale))
        self._reporter = EventReporter(events_path=self.file("events.jsonl"))
        self._previous_reporter = set_reporter(self._reporter)
        self._writer = install_stdout_mirror(self.file("run.log"), self._reporter, mode="a" if self.resume else "w")
        self._t0 = time.time()
        self._manifest = {
            "experiment": self.experiment,
            "dataset": self.dataset,
            "variant": self.variant,
            "seed": self.seed,
            "data_seed": get_settings().run.data_seed,
            "status": "running",
            "started_at": _now(),
            "finished_at": None,
            "duration_seconds": None,
            "command": "evosynth " + shlex.join(sys.argv[1:]),
            "git": git_info(),
            "environment": environment_info(),
            "settings": get_settings().to_dict(),
        }
        atomic_json_dump(self.file("manifest.json"), self._manifest)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            status = "completed"
        elif issubclass(exc_type, KeyboardInterrupt):
            status = "interrupted"
        else:
            status = "failed"
        self._manifest.update(status=status, finished_at=_now(),
                              duration_seconds=round(time.time() - self._t0, 2))
        if exc is not None:
            self._manifest["error"] = f"{exc_type.__name__}: {exc}"
        atomic_json_dump(self.file("manifest.json"), self._manifest)
        sys.stdout = self._writer.original_stdout
        self._writer.close()
        set_reporter(self._previous_reporter)
        self._reporter.close()
        return False

    # -- outputs ----------------------------------------------------------------
    def write_result(self, result):
        """Write result.json: the run's identity followed by `result`."""
        data = {"experiment": self.experiment, "dataset": self.dataset, "variant": self.variant,
                "seed": self.seed, **result}
        atomic_json_dump(self.file("result.json"), data)
        return data
