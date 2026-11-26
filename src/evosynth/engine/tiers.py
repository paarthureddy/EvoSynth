"""
Tier split (Phase 1.6) and elite sharing (Phase 1.9).

TierManager divides the evaluated initial population into Tier 1 islands (one per optimizer)
and the Tier 2 reserve, using one of three methods ([engine] tier_split):

  stratified  (default) every island and every reserve group gets a representative mix of the
              population: one member drawn at random from each equal-width fitness-rank stratum.
              No optimizer starts with an advantage, and Tier 1 is not simply "the best 40%".
  random      a uniform random shuffle, then contiguous slices.
  elite       the v1 behaviour: the best t1_fraction go to Tier 1 in contiguous rank slices
              (GA gets the very best), the worst rest form the reserve. Kept for the split ablation.

share_elites is pairwise: island A's worst members are replaced by B's elites and vice versa.
"""
import numpy as np

from evosynth.seeding import stream

SPLITS = ("stratified", "random", "elite")


def _by_fitness(pop):
    return sorted(pop, key=lambda ind: ind.fitness, reverse=True)


def _stratified_take(sorted_pop, n_take, rng):
    """Pick n_take members: one at random from each of n_take equal-width rank strata."""
    n = len(sorted_pop)
    edges = np.round(np.linspace(0, n, n_take + 1)).astype(int)     # integer strata, sizes differ by <= 1
    picks = [int(rng.integers(lo, hi)) for lo, hi in zip(edges[:-1], edges[1:])]
    chosen = set(picks)
    return [sorted_pop[i] for i in picks], [ind for i, ind in enumerate(sorted_pop) if i not in chosen]


def _deal(members, n_groups, method, rng):
    """Split members into n_groups groups (sizes differ by at most 1)."""
    if method == "stratified":
        members = _by_fitness(members)
        groups = [[] for _ in range(n_groups)]
        for start in range(0, len(members), n_groups):          # one rank block per round
            block = members[start:start + n_groups]
            order = rng.permutation(n_groups)[:len(block)]      # random group for each member
            for ind, g in zip(block, order):
                groups[g].append(ind)
        return groups
    sizes = [len(members) // n_groups + (1 if i < len(members) % n_groups else 0) for i in range(n_groups)]
    out, start = [], 0
    for size in sizes:
        out.append(members[start:start + size])
        start += size
    return out


class TierManager:
    def __init__(self, population, island_size, n_islands=4, split="stratified", rng=None):
        if split not in SPLITS:
            raise ValueError(f"unknown tier split '{split}'. Choose from {SPLITS}")
        t1_count = island_size * n_islands
        if t1_count > len(population):
            raise ValueError(f"population of {len(population)} is too small for {n_islands} islands of {island_size}")
        self.split = split
        self.rng = rng if rng is not None else stream("tiers")
        self.n_islands = n_islands
        self.island_size = island_size

        if split == "stratified":
            t1, reserve = _stratified_take(_by_fitness(population), t1_count, self.rng)
        elif split == "random":
            shuffled = [population[i] for i in self.rng.permutation(len(population))]
            t1, reserve = shuffled[:t1_count], shuffled[t1_count:]
        else:  # elite (v1)
            ranked = _by_fitness(population)
            t1, reserve = ranked[:t1_count], ranked[t1_count:]
        self.tier1_pool = t1
        self.tier2_pool = reserve

    def islands(self):
        """Tier 1 islands, one list per optimizer, each of island_size members."""
        return _deal(self.tier1_pool, self.n_islands, self.split, self.rng)

    def reserve_groups(self, n_groups):
        """The Tier 2 reserve split into n_groups groups (one per promoted optimizer)."""
        return _deal(self.tier2_pool, n_groups, self.split, self.rng)

    @staticmethod
    def select_elites(population, top_pct):
        n = max(1, int(len(population) * top_pct))
        return _by_fitness(population)[:n]

    @staticmethod
    def share_elites(island_a_population, island_b_population, top_pct=0.2):
        """
        Pairwise cooperative co-evolution: island A's worst n are replaced by B's top n and vice
        versa, n = max(1, int(len(smaller island) · top_pct)). Returns the two new populations.
        """
        island_a = _by_fitness(island_a_population)
        island_b = _by_fitness(island_b_population)
        n = max(1, int(min(len(island_a), len(island_b)) * top_pct))
        elites_a, elites_b = island_a[:n], island_b[:n]
        return _by_fitness(island_a[:-n] + elites_b), _by_fitness(island_b[:-n] + elites_a)
