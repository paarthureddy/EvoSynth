"""
Loading, merging and validating the settings folder (settings/*.toml).

Every *.toml file in settings/ is merged into one set of settings; a top-level section
may be defined in only one file. Resolution order (later wins):
    settings/*.toml  →  [experiments.<preset>] engine overrides  →  explicit overrides
    (e.g. from the command line as --set engine.budget=300).

Usage:
    from evosynth.config import load_config
    cfg = load_config()                                   # settings folder only
    cfg = load_config(preset="search_demo")               # + preset (engine keys)
    cfg = load_config(overrides=["engine.budget=300", "bandit.c=0.2"])
    cfg.engine.budget, cfg.composite.fitness, cfg["optimizers"]["pso"]["w"]
    cfg.to_dict()                                         # plain dict, e.g. for manifest.json
"""
from __future__ import annotations

import copy
import math
import tomllib
from pathlib import Path
from typing import Any, Iterable, Mapping

from evosynth.paths import SETTINGS_DIR

# Keys of [engine] that an [experiments.<preset>] table may override directly.
_ENGINE_PRESET_KEYS = {"budget", "pop_size", "t1_iters", "k_iters"}


class ConfigError(ValueError):
    """Raised when the settings file is missing, malformed or holds invalid values."""


class Settings(Mapping):
    """Read-only nested settings with attribute access (cfg.engine.budget)."""

    def __init__(self, data: Mapping[str, Any], path: str = ""):
        object.__setattr__(self, "_data", dict(data))
        object.__setattr__(self, "_path", path)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            where = self._path or "<root>"
            raise AttributeError(f"setting '{name}' not found in [{where}]") from None

    def __setattr__(self, name, value):
        raise AttributeError("Settings are read-only; use load_config(overrides=...)")

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        if isinstance(value, Mapping):
            return Settings(value, f"{self._path}.{key}" if self._path else key)
        return value

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Settings({self._path or 'root'}: {list(self._data)})"


# ---------------------------------------------------------------------------------
# Loading and merging
# ---------------------------------------------------------------------------------
def _deep_merge(base: dict, extra: Mapping) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _parse_override(item: str) -> tuple[list[str], Any]:
    """'engine.budget=300' -> (['engine', 'budget'], 300). Values are parsed as TOML."""
    if "=" not in item:
        raise ConfigError(f"override '{item}' must look like section.key=value")
    dotted, raw = item.split("=", 1)
    keys = [k.strip() for k in dotted.split(".") if k.strip()]
    if not keys:
        raise ConfigError(f"override '{item}' has an empty key")
    raw = raw.strip()
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw  # bare word -> string, e.g. split=stratified
    return keys, value


def _apply_override(data: dict, keys: list[str], value: Any, strict: bool = True) -> None:
    node = data
    for i, key in enumerate(keys[:-1]):
        if key not in node or not isinstance(node[key], dict):
            if strict:
                raise ConfigError(f"unknown setting section '{'.'.join(keys[:i + 1])}'")
            node[key] = {}
        node = node[key]
    if strict and keys[-1] not in node:
        raise ConfigError(f"unknown setting '{'.'.join(keys)}'")
    node[keys[-1]] = value


def read_settings(path: str | Path | None = None) -> dict:
    """
    Raw merged settings. `path` may be the settings folder (default: settings/) or a single
    .toml file. Files in a folder are read in name order; defining the same top-level
    section in two files is an error.
    """
    root = Path(path) if path else SETTINGS_DIR
    if root.is_dir():
        files = sorted(root.glob("*.toml"))
        if not files:
            raise ConfigError(f"no *.toml files in settings folder: {root}")
    elif root.is_file():
        files = [root]
    else:
        raise ConfigError(f"settings folder/file not found: {root}")

    data: dict = {}
    origin: dict[str, str] = {}
    for f in files:
        try:
            with open(f, "rb") as fh:
                part = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{f}: invalid TOML: {exc}") from exc
        for key, value in part.items():
            if key in data:
                raise ConfigError(f"section [{key}] is defined in both {origin[key]} and {f.name}")
            data[key] = value
            origin[key] = f.name
    return data


def load_config(
    path: str | Path | None = None,
    preset: str | None = None,
    overrides: Iterable[str] | Mapping[str, Any] | None = None,
) -> Settings:
    """Load the settings folder (or one .toml file), apply a preset and overrides, validate."""
    data = read_settings(path)

    if preset is not None:
        presets = data.get("experiments", {})
        if preset not in presets:
            raise ConfigError(f"unknown preset '{preset}'. Available: {sorted(presets)}")
        engine_part = {k: v for k, v in presets[preset].items() if k in _ENGINE_PRESET_KEYS}
        data["engine"] = _deep_merge(data["engine"], engine_part)

    if overrides:
        items = overrides.items() if isinstance(overrides, Mapping) else (
            _parse_override(o) for o in overrides)
        for keys, value in items:
            keys = keys.split(".") if isinstance(keys, str) else keys
            _apply_override(data, keys, value)

    validate(data)
    return Settings(data)


# ---------------------------------------------------------------------------------
# Active settings (what library code reads its defaults from)
# ---------------------------------------------------------------------------------
_ACTIVE: Settings | None = None


def get_settings() -> Settings:
    """Settings currently in force. Loads the settings/ folder on first use."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = load_config()
    return _ACTIVE


def set_settings(settings: Settings) -> None:
    """Make `settings` (e.g. load_config(preset=..., overrides=...)) the active settings.
    Call this before constructing optimizers/evaluators for a run."""
    global _ACTIVE
    if not isinstance(settings, Settings):
        raise TypeError("set_settings() expects the result of load_config()")
    _ACTIVE = settings


# ---------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------
def _require(cond: bool, msg: str, errors: list[str]) -> None:
    if not cond:
        errors.append(msg)


def _get(data: Mapping, dotted: str, errors: list[str]) -> Any:
    node: Any = data
    for key in dotted.split("."):
        if not isinstance(node, Mapping) or key not in node:
            errors.append(f"missing setting '{dotted}'")
            return None
        node = node[key]
    return node


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_range_table(name: str, rng: Any, errors: list[str]) -> None:
    if not isinstance(rng, Mapping) or "low" not in rng or "high" not in rng:
        errors.append(f"{name} must be a table {{ low = .., high = .. }}")
        return
    lo, hi = rng["low"], rng["high"]
    _require(_is_num(lo) and _is_num(hi) and lo < hi, f"{name}: need low < high (got {lo}, {hi})", errors)
    if rng.get("log"):
        _require(_is_num(lo) and lo > 0, f"{name}: log scale needs low > 0", errors)


def validate(data: Mapping) -> None:
    """Check types, ranges and cross-setting rules. Raises ConfigError listing every problem."""
    e: list[str] = []

    def pos_int(key, minimum=1):
        v = _get(data, key, e)
        _require(isinstance(v, int) and not isinstance(v, bool) and v >= minimum,
                 f"{key} must be an integer >= {minimum} (got {v!r})", e)

    def frac(key, lo=0.0, hi=1.0):
        v = _get(data, key, e)
        _require(_is_num(v) and lo <= v <= hi, f"{key} must be a number in [{lo}, {hi}] (got {v!r})", e)

    # run
    pos_int("run.seed", 0)
    pos_int("run.data_seed", 0)
    pos_int("run.max_workers", 0)

    # engine
    for k in ("budget", "pop_size", "t1_iters", "k_iters"):
        pos_int(f"engine.{k}")
    frac("engine.t1_fraction", 0.05, 0.95)
    n_prom = _get(data, "engine.n_promoted", e)
    _require(n_prom == 2, "engine.n_promoted: only 2 is supported until the tier logic is generalised", e)
    pos_int("engine.min_island_size", 2)
    pos_int("engine.min_t2_checkpoints", 1)
    _require(isinstance(_get(data, "engine.auto_k", e), bool), "engine.auto_k must be true or false", e)
    split = _get(data, "engine.tier_split", e)
    _require(split in ("stratified", "random", "elite"),
             f"engine.tier_split must be 'stratified', 'random' or 'elite' (got {split!r})", e)
    pos_int("composite.trend_window", 2)
    _require(_is_num(_get(data, "telemetry.reprobe_seconds", e)) and _get(data, "telemetry.reprobe_seconds", e) >= 0,
             "telemetry.reprobe_seconds must be >= 0", e)
    pop = _get(data, "engine.pop_size", e)
    t1f = _get(data, "engine.t1_fraction", e)
    min_island = data.get("engine", {}).get("min_island_size", 4)
    if _is_num(t1f) and isinstance(min_island, int):
        sized = [("engine.pop_size", pop)] + [
            (f"experiments.{name}.pop_size", preset.get("pop_size"))
            for name, preset in data.get("experiments", {}).items() if isinstance(preset, dict)]
        for key, value in sized:
            if isinstance(value, int):
                _require(int(value * t1f) // 4 >= min_island,
                         f"{key}={value} x engine.t1_fraction={t1f} gives Tier 1 islands of "
                         f"{int(value * t1f) // 4}; at least engine.min_island_size={min_island} are needed "
                         f"(pop_size >= {-(-min_island * 4 // t1f):.0f})", e)
    budget = _get(data, "engine.budget", e)
    if isinstance(pop, int) and isinstance(budget, int):
        _require(budget > pop, f"engine.budget ({budget}) must exceed engine.pop_size ({pop})", e)

    # bandit
    c = _get(data, "bandit.c", e)
    _require(_is_num(c) and c >= 0, f"bandit.c must be >= 0 (got {c!r})", e)

    # composite weights
    names = ("fitness", "speed", "efficiency", "diversity", "trend")
    weights = [_get(data, f"composite.{n}", e) for n in names]
    for n, w in zip(names, weights):
        _require(_is_num(w) and 0 <= w <= 1, f"composite.{n} must be in [0, 1] (got {w!r})", e)
    if all(_is_num(w) for w in weights):
        total = sum(weights)
        _require(math.isclose(total, 1.0, abs_tol=1e-6),
                 f"composite weights must sum to 1.0 (got {total:.4f})", e)
    frac("composite.tie_value")

    # safeguards
    frac("safeguards.swap_margin", 0.0, 1.0)
    pos_int("safeguards.grace_iters")
    pos_int("safeguards.cooldown_multiplier", 0)
    pos_int("safeguards.stagnation_limit")
    for k in ("t1_inject_pct", "t2_inject_pct"):
        frac(f"safeguards.{k}")
    frac("safeguards.elite_pct", 0.0, 0.5)

    # optimizers
    frac("optimizers.ga.mutation_rate")
    v = _get(data, "optimizers.ga.mutation_sigma", e)
    _require(_is_num(v) and v > 0, "optimizers.ga.mutation_sigma must be > 0", e)
    pos_int("optimizers.ga.tournament_size")
    for k in ("w", "c1", "c2", "init_velocity"):
        v = _get(data, f"optimizers.pso.{k}", e)
        _require(_is_num(v) and v >= 0, f"optimizers.pso.{k} must be >= 0", e)
    v = _get(data, "optimizers.de.F", e)
    _require(_is_num(v) and 0 < v <= 2, "optimizers.de.F must be in (0, 2]", e)
    frac("optimizers.de.CR")
    v = _get(data, "optimizers.cmaes.sigma0", e)
    _require(_is_num(v) and v > 0, "optimizers.cmaes.sigma0 must be > 0", e)
    pos_int("optimizers.cmaes.min_popsize", 2)
    frac("optimizers.cmaes.default_mean")

    # search spaces
    tab = _get(data, "search_space.tabular", e) or {}
    for k in ("learning_rate", "n_estimators", "max_depth", "subsample", "colsample_bytree", "reg_alpha"):
        if k in tab:
            _check_range_table(f"search_space.tabular.{k}", tab[k], e)
        else:
            e.append(f"missing setting 'search_space.tabular.{k}'")
    for k in ("model_types", "fe_strategies", "boosters"):
        _require(isinstance(tab.get(k), list) and len(tab.get(k)) >= 1,
                 f"search_space.tabular.{k} must be a non-empty list", e)
    allowed_models = {"XGBoost", "RandomForest", "LightGBM", "ExtraTrees"}
    _require(set(tab.get("model_types", [])) <= allowed_models,
             f"search_space.tabular.model_types may only contain {sorted(allowed_models)}", e)
    _require(set(tab.get("fe_strategies", [])) <= {"none", "stats", "interactions"},
             "search_space.tabular.fe_strategies may only contain none / stats / interactions", e)

    # evaluation / features / refit
    pos_int("evaluation.cv_folds", 2)
    pos_int("evaluation.early_stopping_rounds")
    pos_int("evaluation.search_tree_cap")
    pos_int("evaluation.bagging_depth_multiplier")
    frac("evaluation.failed_fold_score")
    pos_int("features.continuous_unique_threshold", 2)
    pos_int("features.detection_sample_rows", 10)
    pos_int("features.max_interaction_features", 2)
    for k in ("train_rows", "test_rows", "min_trees_bagging", "min_trees_lgbm", "min_trees_xgb"):
        pos_int(f"refit.{k}")

    # datasets
    ds = _get(data, "datasets", e) or {}
    _require(isinstance(ds, Mapping) and len(ds) >= 1, "at least one [datasets.<name>] is required", e)
    for name, spec in (ds.items() if isinstance(ds, Mapping) else []):
        _require(isinstance(spec, Mapping) and "source" in spec, f"datasets.{name}.source is required", e)

    # telemetry
    pos_int("telemetry.server_port", 1)
    for k in ("probe_timeout", "post_timeout"):
        v = _get(data, f"telemetry.{k}", e)
        _require(_is_num(v) and v > 0, f"telemetry.{k} must be > 0", e)

    # experiment presets: engine keys must be valid where present
    for name, preset in (_get(data, "experiments", e) or {}).items():
        for k in _ENGINE_PRESET_KEYS & set(preset):
            v = preset[k]
            _require(isinstance(v, int) and v >= 1, f"experiments.{name}.{k} must be an integer >= 1", e)
        if {"budget", "pop_size"} <= set(preset):
            _require(preset["budget"] > preset["pop_size"],
                     f"experiments.{name}: budget must exceed pop_size", e)

    if e:
        raise ConfigError("invalid settings:\n  - " + "\n  - ".join(e))
