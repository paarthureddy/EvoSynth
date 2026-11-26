"""
UCB1 multi-armed bandit over the optimizer families (Phase 1.1: it makes the decisions).

Each optimizer family (GA, PSO, DE, CMAES) is an arm. Its reward is the composite score from
CompositeScoringEngine; the bandit adds the UCB1 exploration bonus

    UCB(arm) = composite(arm) + c · sqrt( ln(N) / n_arm )

with N = total evaluations so far and n_arm = evaluations spent on that family. Promotion
(Step 4) and SWAP (Step 5) rank the arms by this score.

The composite score is re-normalised over the islands being compared at every scoring call, so
the exploitation term is the current composite, not a running mean across calls; the running
mean of rewards is still recorded (update) for reporting.

Settings: [bandit] c in settings/engine.toml.
"""
import math

from evosynth.config import get_settings


class UCB1Bandit:
    def __init__(self, c=None):
        self.c = get_settings().bandit.c if c is None else c
        self.total_pulls = 0
        self.stats = {}  # arm -> {'pulls': int, 'total_reward': float}

    def register_algorithm(self, algo_name):
        if algo_name not in self.stats:
            self.stats[algo_name] = {'pulls': 0, 'total_reward': 0.0}

    def update(self, algo_name, reward):
        """Record one scoring call's composite score (reward) for an arm."""
        self.register_algorithm(algo_name)
        self.stats[algo_name]['pulls'] += 1
        self.stats[algo_name]['total_reward'] += reward
        self.total_pulls += 1

    def mean_reward(self, algo_name):
        st = self.stats.get(algo_name)
        return st['total_reward'] / st['pulls'] if st and st['pulls'] else 0.0

    def bonus(self, n_algo, n_total):
        """Exploration bonus c · sqrt(ln N / n): larger for families that have had fewer evaluations."""
        return self.c * math.sqrt(math.log(max(1, n_total)) / max(1, n_algo))

    def score(self, algo_name, composite, n_algo, n_total):
        """UCB score used for promotion and SWAP decisions."""
        return composite + self.bonus(n_algo, n_total)

    def rank(self, ucb_scores):
        """Arm names sorted by UCB score, best first (ties keep insertion order)."""
        return sorted(ucb_scores, key=lambda k: ucb_scores[k], reverse=True)

    def select_next(self, available_algorithms):
        """Classic UCB1 on mean rewards (kept for completeness; the controller uses score/rank)."""
        for algo in available_algorithms:
            self.register_algorithm(algo)
            if self.stats[algo]['pulls'] == 0:
                return algo
        return max(available_algorithms,
                   key=lambda a: self.mean_reward(a) + self.bonus(self.stats[a]['pulls'], self.total_pulls))
