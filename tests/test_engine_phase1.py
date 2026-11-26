"""Phase 1: bandit decisions, metrics, tier split, auto k, inject(), pairwise elite sharing."""
import math

import numpy as np
import pytest

from evosynth.config import load_config, set_settings
from evosynth.core.individual import Individual
from evosynth.engine.bandit import UCB1Bandit
from evosynth.engine.controller import DynamicOptimizer
from evosynth.engine.tiers import TierManager
from evosynth.optimizers.cmaes import CMAESOptimizer
from evosynth.optimizers.pso import ParticleSwarm
from evosynth.search_spaces.toy import get_dummy_search_space
from evosynth.seeding import stream


def _pop(n, dims=4, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ind = Individual(rng.random(dims))
        ind.set_fitness(float(i))            # fitness = rank, so strata are easy to check
        out.append(ind)
    return out


# ---------------------------------------------------------------- 1.1 bandit
def test_bandit_uses_settings_c_and_ucb_formula():
    b = UCB1Bandit()
    assert b.c == pytest.approx(0.1)
    assert b.score("GA", 0.5, n_algo=10, n_total=100) == pytest.approx(0.5 + 0.1 * math.sqrt(math.log(100) / 10))
    assert b.bonus(5, 100) > b.bonus(50, 100)          # under-used families get a larger bonus
    assert b.rank({"A": 0.2, "B": 0.9, "C": 0.5}) == ["B", "C", "A"]


def test_controller_records_every_scoring_call_in_the_bandit(toy_problem):
    space, ev = toy_problem
    opt = DynamicOptimizer(space, ev, budget=200, pop_size=40, t1_iters=2, k_iters=1, max_workers=1)
    opt.run()
    assert opt.bandit.total_pulls >= 8                 # 4 T1 families per scoring call, several calls
    assert set(opt.bandit.stats) == {"GA", "PSO", "DE", "CMAES"}


# ---------------------------------------------------------------- 1.2-1.4 metrics
def test_latent_diversity():
    same = [Individual(np.full(4, 0.5)) for _ in range(5)]
    assert DynamicOptimizer._latent_diversity(same, 4) == 0.0
    corners = [Individual(np.zeros(4)), Individual(np.ones(4))]
    assert DynamicOptimizer._latent_diversity(corners, 4) == pytest.approx(1.0)
    assert 0.0 < DynamicOptimizer._latent_diversity(_pop(10), 4) < 1.0


def test_trend_is_a_slope_and_differs_from_speed(toy_problem):
    space, ev = toy_problem
    opt = DynamicOptimizer(space, ev, budget=200, pop_size=40, max_workers=1)
    assert opt._trend([80.0]) == 0.0
    assert opt._trend([80.0, 81.0, 82.0]) == pytest.approx(1.0)
    assert opt._trend([70.0, 80.0, 81.0, 82.0]) == pytest.approx(1.0)   # only the last 3 calls count


def test_metrics_in_a_scoring_call(toy_problem):
    space, ev = toy_problem
    opt = DynamicOptimizer(space, ev, budget=200, pop_size=40, max_workers=1)
    pops = {"T1_GA": _pop(4, seed=1), "T1_PSO": _pop(4, seed=2)}
    opt._algo_stats["GA"]["eval_seconds"] = 2.0
    opt._algo_stats["PSO"]["eval_seconds"] = 1.0
    _, raw = opt._compute_composite(pops)
    assert raw["T1_GA"]["speed"] == pytest.approx(3.0)              # best 3 - previous 0
    assert raw["T1_GA"]["efficiency"] == pytest.approx(1.5)         # 3 / 2 s
    assert raw["T1_PSO"]["efficiency"] == pytest.approx(3.0)        # 3 / 1 s
    assert 0 < raw["T1_GA"]["diversity"] < 1


def test_small_islands_are_rejected(toy_problem):
    space, ev = toy_problem
    with pytest.raises(ValueError, match="min_island_size"):
        DynamicOptimizer(space, ev, budget=200, pop_size=20)


# ---------------------------------------------------------------- 1.6 / 1.9 tiers
def test_stratified_split_gives_every_island_the_whole_range():
    pop = _pop(40)
    tm = TierManager(pop, island_size=4, split="stratified", rng=stream("t", seed=1))
    islands = tm.islands()
    assert [len(i) for i in islands] == [4, 4, 4, 4]
    assert len(tm.tier2_pool) == 24
    ranks = [sorted(int(ind.fitness) for ind in isl) for isl in islands]
    for r in ranks:                                    # each island spans good and bad members
        assert max(r) - min(r) >= 15
    t1 = [int(i.fitness) for i in tm.tier1_pool]
    assert max(t1) - min(t1) >= 30                     # Tier 1 is not just the top 40%
    groups = tm.reserve_groups(2)
    assert [len(g) for g in groups] == [12, 12]
    ids = [id(i) for isl in islands for i in isl] + [id(i) for g in groups for i in g]
    assert len(set(ids)) == 40                         # nobody lost, nobody duplicated


def test_elite_split_reproduces_v1():
    tm = TierManager(_pop(40), island_size=4, split="elite", rng=stream("t", seed=1))
    islands = tm.islands()
    assert [int(i.fitness) for i in islands[0]] == [39, 38, 37, 36]   # GA gets the very best
    assert max(int(i.fitness) for i in tm.tier2_pool) == 23


def test_split_is_reproducible_and_seed_dependent():
    a = [[int(i.fitness) for i in isl] for isl in TierManager(_pop(40), 4, split="random", rng=stream("t", seed=3)).islands()]
    b = [[int(i.fitness) for i in isl] for isl in TierManager(_pop(40), 4, split="random", rng=stream("t", seed=3)).islands()]
    c = [[int(i.fitness) for i in isl] for isl in TierManager(_pop(40), 4, split="random", rng=stream("t", seed=4)).islands()]
    assert a == b and a != c


def test_pairwise_share_elites():
    a, b = _pop(10, seed=1), _pop(10, seed=2)
    for ind in b:
        ind.fitness += 100                             # island B is much better
    new_a, new_b = TierManager.share_elites(a, b, top_pct=0.2)
    assert len(new_a) == 10 and len(new_b) == 10
    assert sorted(i.fitness for i in new_a)[-2:] == [108, 109]   # A received B's top 2
    assert min(i.fitness for i in new_a) == 2                    # A's worst 2 (0, 1) were dropped
    assert {9.0, 8.0} <= {i.fitness for i in new_b}              # B received A's top 2


# ---------------------------------------------------------------- 1.7 auto k
@pytest.mark.parametrize("budget, k_conf, expected", [(200, 10, 1), (500, 10, 5), (2000, 10, 10)])
def test_auto_k_gives_tier2_two_checkpoints(toy_problem, budget, k_conf, expected):
    space, ev = toy_problem
    opt = DynamicOptimizer(space, ev, budget=budget, pop_size=40, t1_iters=3, k_iters=k_conf, max_workers=1)
    opt._set_checkpoint_interval(island_size=4)
    assert opt.k == expected
    t2_iters = (budget - 40 - 3 * 16) // 40
    assert t2_iters // opt.k >= 2 or opt.k == 1


def test_auto_k_can_be_switched_off(toy_problem):
    set_settings(load_config(overrides=["engine.auto_k=false"]))
    space, ev = toy_problem
    opt = DynamicOptimizer(space, ev, budget=200, pop_size=40, k_iters=7, max_workers=1)
    opt._set_checkpoint_interval(island_size=4)
    assert opt.k == 7


# ---------------------------------------------------------------- 1.8 inject
def test_inject_replaces_worst_members():
    space = get_dummy_search_space()
    pso = ParticleSwarm(space, 6, rng=stream("t", seed=1))
    pso.tell(_pop(6))
    pso.ask(6)                                         # builds the particles from the seed population
    star = Individual(np.full(4, 0.25)); star.set_fitness(1000.0)
    pso.inject([star])
    assert pso.population[0] is star and len(pso.population) == 6
    assert pso.gbest_score == 1000.0 and np.allclose(pso.gbest_pos, 0.25)
    assert sum(p["pbest_score"] == 1000.0 for p in pso.particles) == 1


def test_cmaes_restarts_around_new_mean_after_inject():
    space = get_dummy_search_space()
    cma_opt = CMAESOptimizer(space, 8, rng=stream("t", seed=1))
    cma_opt.tell(_pop(8))
    cma_opt.ask(8)
    assert cma_opt.es is not None
    cma_opt.inject([_pop(1, seed=9)[0]])
    assert cma_opt.es is None                          # next ask starts a new distribution
    assert len(cma_opt.ask(8)) == 8


def test_full_run_with_every_split_method(toy_problem):
    space, ev = toy_problem
    for split in ("stratified", "random", "elite"):
        set_settings(load_config(overrides=[f'engine.tier_split="{split}"']))
        opt = DynamicOptimizer(space, ev, budget=200, pop_size=40, t1_iters=2, k_iters=1, max_workers=1)
        best = opt.run()
        assert opt.evals_used == 200 and best.fitness > 0


def test_cmaes_updates_its_distribution_without_errors(capsys):
    # seed population (not sampled by cma) followed by several full generations
    space = get_dummy_search_space()
    opt = CMAESOptimizer(space, 4, rng=stream("t", seed=2))
    opt.tell(_pop(4))
    for gen in range(5):
        vecs = opt.ask(4)
        inds = []
        for v in vecs:
            ind = Individual(v); ind.set_fitness(float(-np.sum((v - 0.3) ** 2))); inds.append(ind)
        opt.tell(inds)
    assert opt.n_restarts == 0 and "[CMA-ES] update failed" not in capsys.readouterr().out
    assert opt.es.countiter == 5                        # one update per sampled generation
