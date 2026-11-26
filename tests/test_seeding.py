"""Phase 0: one seed reproduces a run exactly; different seeds differ; nothing reads the clock."""
import numpy as np
import pytest

from evosynth.cli import _seed_list, build_parser
from evosynth.config import ConfigError, load_config, set_settings
from evosynth.engine.controller import DynamicOptimizer
from evosynth.optimizers.cmaes import CMAESOptimizer
from evosynth.search_spaces.tabular import get_tabular_search_space
from evosynth.seeding import derive_int, stream


def _toy_run(space, evaluator, seed):
    set_settings(load_config(overrides=[f"run.seed={seed}"]))
    opt = DynamicOptimizer(space, evaluator, budget=80, pop_size=20, t1_iters=2, k_iters=1, max_workers=4)
    best = opt.run()
    return [(h["eval_id"], round(h["fitness"], 10), h["algo"]) for h in opt.history], best


def test_same_seed_reproduces_the_whole_run(toy_problem):
    # includes CMA-ES, which used to reseed numpy from the clock on every creation
    h1, b1 = _toy_run(*toy_problem, seed=7)
    np.random.seed(None)                      # scramble the global generators in between
    np.random.rand(1000)
    h2, b2 = _toy_run(*toy_problem, seed=7)
    assert h1 == h2
    assert np.array_equal(b1.vector, b2.vector) and b1.source_algorithm == b2.source_algorithm


def test_different_seeds_differ(toy_problem):
    h1, _ = _toy_run(*toy_problem, seed=1)
    h2, _ = _toy_run(*toy_problem, seed=2)
    assert h1 != h2


def test_cmaes_samples_from_its_own_generator():
    space = get_tabular_search_space()
    np.random.seed(123)
    before = np.random.get_state()[1][:5].copy()
    a = CMAESOptimizer(space, 8, rng=stream("t", seed=5)).ask(8)
    b = CMAESOptimizer(space, 8, rng=stream("t", seed=5)).ask(8)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert np.array_equal(np.random.get_state()[1][:5], before)   # global numpy left alone


def test_streams_are_independent_and_reproducible():
    a1 = stream("optimizer", "GA", seed=3).random(5)
    noisy = stream("optimizer", "PSO", seed=3)
    noisy.random(10_000)                       # draws elsewhere don't shift GA's numbers
    a2 = stream("optimizer", "GA", seed=3).random(5)
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, stream("optimizer", "GA", seed=4).random(5))
    assert derive_int("x", seed=1) == derive_int("x", seed=1) != derive_int("x", seed=2)


def test_seed_and_data_seed_are_separate_settings():
    s = load_config(overrides=["run.seed=5"])
    assert s.run.seed == 5 and s.run.data_seed == 42
    with pytest.raises(ConfigError):
        load_config(overrides=["run.seed=-1"])


def test_seed_flags():
    p = build_parser()
    assert _seed_list(p.parse_args(["run"])) == [None]
    assert _seed_list(p.parse_args(["run", "--seed", "9"])) == [9]
    assert _seed_list(p.parse_args(["scale", "--seeds", "1, 2,3"])) == [1, 2, 3]
    with pytest.raises(ConfigError):
        _seed_list(p.parse_args(["run", "--seeds", "1,x"]))
    with pytest.raises(SystemExit):
        p.parse_args(["run", "--seed", "1", "--seeds", "2,3"])
