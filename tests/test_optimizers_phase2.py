"""Phase 2: PSO never re-proposes its seed population; DE uses one-to-one selection; both cycle."""
import numpy as np

from evosynth.core.individual import Individual
from evosynth.optimizers.de import DifferentialEvolution
from evosynth.optimizers.pso import ParticleSwarm
from evosynth.search_spaces.toy import get_dummy_search_space
from evosynth.seeding import stream


def _pop(n, seed=0, dims=4):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ind = Individual(rng.random(dims))
        ind.set_fitness(float(i))
        out.append(ind)
    return out


def _evaluate(vectors, fitness):
    out = []
    for v in vectors:
        ind = Individual(v)
        ind.set_fitness(fitness(v))
        out.append(ind)
    return out


def test_pso_first_ask_moves_away_from_the_seed_population():
    pso = ParticleSwarm(get_dummy_search_space(), 8, rng=stream("t", seed=1))
    seeds = _pop(8)
    pso.tell(seeds)
    proposed = pso.ask(8)
    for v in proposed:
        assert not any(np.array_equal(v, s.vector) for s in seeds)


def test_pso_moves_every_particle_with_small_batches():
    pso = ParticleSwarm(get_dummy_search_space(), 12, rng=stream("t", seed=1))
    pso.tell(_pop(12))
    for _ in range(3):                                       # 3 asks of 4 = the whole swarm once
        pso.tell(_evaluate(pso.ask(4), lambda v: -float(np.sum(v))))
    start = [s.vector for s in _pop(12)]
    moved = [not np.array_equal(p["pos"], s) for p, s in zip(pso.particles, start)]
    assert all(moved)


def test_pso_tell_updates_the_particle_that_proposed_the_vector():
    pso = ParticleSwarm(get_dummy_search_space(), 6, rng=stream("t", seed=2))
    pso.tell(_pop(6))
    pso.tell(_evaluate(pso.ask(3), lambda v: 0.0))           # particles 0-2
    vecs = pso.ask(3)                                        # particles 3-5
    pso.tell(_evaluate(vecs, lambda v: 1000.0))
    assert [p["pbest_score"] == 1000.0 for p in pso.particles] == [False] * 3 + [True] * 3


def test_de_one_to_one_selection():
    de = DifferentialEvolution(get_dummy_search_space(), 6, rng=stream("t", seed=3))
    de.tell(_pop(6))                                         # fitness 0..5
    before = [ind.fitness for ind in de.population]
    trials = de.ask(6)
    # every trial scores 2.5: it replaces only targets worse than itself, one for one
    de.tell(_evaluate(trials, lambda v: 2.5))
    after = [ind.fitness for ind in de.population]
    assert after == [max(b, 2.5) for b in before]
    assert len(de.population) == 6


def test_de_cycles_through_all_targets():
    de = DifferentialEvolution(get_dummy_search_space(), 8, rng=stream("t", seed=4))
    de.tell(_pop(8))
    seen = []
    for _ in range(4):
        de.ask(2)
        seen += de._targets
        de.tell(_evaluate([np.zeros(4)] * 2, lambda v: -1.0))
    assert sorted(seen) == list(range(8))


def test_de_keeps_the_best_and_converges_on_a_sphere():
    de = DifferentialEvolution(get_dummy_search_space(), 10, rng=stream("t", seed=5))
    f = lambda v: -float(np.sum((v - 0.3) ** 2))
    de.tell(_evaluate([de.rng.random(4) for _ in range(10)], f))
    best0 = max(i.fitness for i in de.population)
    for _ in range(30):
        de.tell(_evaluate(de.ask(10), f))
    best = max(i.fitness for i in de.population)
    assert best >= best0 and best > -0.01
