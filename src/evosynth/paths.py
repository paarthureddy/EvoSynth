"""
Central filesystem locations for EvoSynth.

Every module that needs the project root, the dataset cache or the results
folder imports it from here instead of computing it from its own __file__.
Paths assume the package is installed in editable mode from the repository
(`pip install -e .`), i.e. this file lives at <repo>/src/evosynth/paths.py.
"""
from pathlib import Path

# <repo>/src/evosynth/paths.py -> parents[0]=evosynth, [1]=src, [2]=<repo>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Local dataset cache (git-ignored). sklearn/OpenML/torchvision downloads land here.
DATA_DIR = PROJECT_ROOT / ".data"

# Default output folder for engine logs and population CSVs.
RESULTS_DIR = PROJECT_ROOT / "results"

# Master settings file (see configs/evosynth.toml).
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_FILE = CONFIG_DIR / "evosynth.toml"
