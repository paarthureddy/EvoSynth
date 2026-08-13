import threading
# Injected by server.py to allow stopping from the UI
_STOP_FLAG: threading.Event = threading.Event()
_STOP_FLAG.clear()   # starts cleared (not stopped)

"""
Hybrid AutoML Strategy Switching — Full PDF-spec implementation.
Phases:
  1  Initialization     : N=50 random individuals, random tier assignment
  2  T1 Warm-up         : GA/PSO/DE/CMA-ES each get 5 inds for 5 iters
  3  Composite scoring  : weights Acc=0.30 Speed=0.25 Eff=0.15 Div=0.15 Trend=0.15
  4  T2 Promotion       : top-2 → T2 (15 each), bottom-2 stay T1
  5  Parallel main loop : all 6 islands evolve simultaneously
  6  Every k iters      : T2 island sharing + challenge check + stagnation check
  7  Every 10 iters     : T2→T1 top-down knowledge injection
  8  Convergence        : budget exhausted → return best + ensemble top-5

After every step the full system state is written to ui/state.json so the
live dashboard can visualise the run in real-time.
"""

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

from automl_engine.core.individual import Individual
from automl_engine.engine.composite import CompositeScoringEngine
from automl_engine.engine.ucb1 import UCB1Bandit
from automl_engine.optimizers.cmaes import CMAES
from automl_engine.optimizers.de import DifferentialEvolution
from automl_engine.optimizers.ga import GeneticAlgorithm
from automl_engine.optimizers.pso import ParticleSwarm

# ── State-file path (relative to this file → ui/ folder) ─────────────────────
_UI_DIR = Path(__file__).resolve().parents[3] / "ui"
STATE_FILE = _UI_DIR / "state.json"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# ── Pretty-print helpers ──────────────────────────────────────────────────────
def _sep(c="=", w=72): return c * w
def _hdr(t, c="=", w=72):
    s = (w - len(t) - 2) // 2
    return f"{c*s} {t} {c*s}"


# ── Optimizer factory ─────────────────────────────────────────────────────────
def _make_opt(key, space, pop_size):
    m = {"GA": GeneticAlgorithm, "PSO": ParticleSwarm,
         "DE": DifferentialEvolution, "CMAES": CMAES}
    return m[key](space, pop_size)


# ── Vector → human-readable hyperparameters ───────────────────────────────────
def _decode(individual, space):
    import ConfigSpace.hyperparameters as CSH
    from ConfigSpace import Configuration
    v = np.copy(individual.vector)
    for i, name in enumerate(list(space.keys())):
        hp = space.get_hyperparameter(name)
        if isinstance(hp, CSH.CategoricalHyperparameter):
            n = len(hp.choices)
            v[i] = float(min(int(np.floor(v[i] * n)), n - 1))
    cfg = Configuration(space, vector=v)
    return {p: cfg[p] for p in list(space.keys())}


# ── Serialise an optimizer's population for JSON ──────────────────────────────
def _pop_json(opt, tag):
    out = []
    for ind in getattr(opt, "population", []):
        if ind.fitness is not None:
            out.append({"fitness": round(float(ind.fitness), 4),
                        "src": ind.source_algorithm or tag})
    out.sort(key=lambda x: x["fitness"], reverse=True)
    return out


# ── Main controller ───────────────────────────────────────────────────────────
class DynamicOptimizer:
    ALL_ALGOS = ["GA", "PSO", "DE", "CMAES"]
    WEIGHTS   = dict(fitness=0.30, speed=0.25, efficiency=0.15,
                     diversity=0.15, trend=0.15)

    # Warm-up composite weights match the PDF composite engine weights
    def __init__(self, search_space, evaluator,
                 budget=300, k_iters=5,
                 step_delay=0.08,
                 swap_threshold=0.05,
                 stagnation_limit=2,
                 elite_pct=0.20,
                 inject_pct=0.20):
        self.space       = search_space
        self.eval        = evaluator
        self.budget      = budget
        self.k           = k_iters
        self.swap_thr    = swap_threshold
        self.stag_limit  = stagnation_limit
        self.elite_pct   = elite_pct
        self.inject_pct  = inject_pct
        self.dims        = len(list(search_space.keys()))

        self.step_delay = step_delay
        self.composite_engine = CompositeScoringEngine()
        self.bandit           = UCB1Bandit()
        for a in self.ALL_ALGOS:
            self.bandit.register_algorithm(a)

        self.evals_used      = 0
        self.best_individual = None
        self.all_individuals = []   # every evaluated ind for ensemble

        # Per-algorithm stat tracking for composite scoring
        self._stats = {a: {"best": 0.0, "prev_best": 0.0, "history": []}
                       for a in self.ALL_ALGOS}

        # Runtime state (written to state.json every iteration)
        self._phase      = "INIT"
        self._iteration  = 0
        self._events     = []         # last N events for event log
        self._start_time = time.time()

    # ── Evaluation ───────────────────────────────────────────────────────────
    def _eval_batch(self, vectors, tag):
        key = tag.split("_")[-1]
        inds = []
        for v in vectors:
            if self.evals_used >= self.budget:
                break
            ind = Individual(v)
            ind.set_fitness(self.eval.evaluate(ind))
            ind.source_algorithm = tag
            inds.append(ind)
            self.all_individuals.append(ind)
            self.evals_used += 1
            if key in self._stats and ind.fitness > self._stats[key]["best"]:
                self._stats[key]["best"] = ind.fitness
            if (self.best_individual is None or
                    ind.fitness > self.best_individual.fitness):
                self.best_individual = ind
        return inds

    # ── Composite scoring ─────────────────────────────────────────────────────
    def _composite(self, named_pops):
        """
        named_pops: dict[label → list[Individual]]
        returns (scores_dict, raw_dict)
        """
        raw = {}
        for label, pop in named_pops.items():
            if not pop:
                continue
            fits = [i.fitness for i in pop if i.fitness is not None]
            if not fits:
                continue
            key  = label.split("_")[-1]
            best = max(fits)
            prev = self._stats[key]["prev_best"]
            delta = max(best - prev, 0.0)
            spread = max(fits) - min(fits) + 1e-9
            div    = float(np.std(fits)) / spread if len(fits) > 1 else 0.0
            div    = min(div, 1.0)
            h = self._stats[key]["history"]
            trend = (h[-1] - h[-3]) / 2.0 if len(h) >= 3 else delta
            self._stats[key]["prev_best"] = best
            self._stats[key]["history"].append(best)
            raw[label] = dict(fitness=best, speed=delta,
                              efficiency=delta / self.k,
                              diversity=div, trend=trend)
        scores = self.composite_engine.calculate_scores(raw)
        return scores, raw

    # ── Event log ────────────────────────────────────────────────────────────
    def _event(self, msg, kind="INFO"):
        ts = round(time.time() - self._start_time, 1)
        entry = {"t": ts, "kind": kind, "msg": msg}
        self._events.append(entry)
        if len(self._events) > 100:
            self._events = self._events[-100:]
        print(f"  [{kind}] {msg}")

    # ── Write state.json ─────────────────────────────────────────────────────
    def _dump_state(self, t1_optims, t2_optims, t1_scores, t2_scores, raw_all):
        best_params = {}
        if self.best_individual:
            try:
                best_params = _decode(self.best_individual, self.space)
            except Exception:
                pass

        # Build per-algorithm composite table for UI
        composite_table = {}
        for label, sc in {**t1_scores, **t2_scores}.items():
            key = label.split("_")[-1]
            r   = raw_all.get(label, {})
            composite_table[key] = {
                "tier":       "T1" if label.startswith("T1") else "T2",
                "composite":  round(sc, 4),
                "accuracy":   round(r.get("fitness", 0), 4),
                "speed":      round(r.get("speed", 0), 4),
                "diversity":  round(r.get("diversity", 0), 4),
                "trend":      round(r.get("trend", 0), 4),
            }

        # T1 / T2 island populations
        t1_islands = {k: _pop_json(v, f"T1_{k}") for k, v in t1_optims.items()}
        t2_islands = {k: _pop_json(v, f"T2_{k}") for k, v in t2_optims.items()}

        state = {
            "phase":       self._phase,
            "iteration":   self._iteration,
            "evals_used":  self.evals_used,
            "budget":      self.budget,
            "elapsed":     round(time.time() - self._start_time, 1),
            "best_fitness": round(self.best_individual.fitness, 4)
                             if self.best_individual else 0,
            "best_algo":   self.best_individual.source_algorithm
                             if self.best_individual else "",
            "best_params": {k: (round(v, 6) if isinstance(v, float) else v)
                            for k, v in best_params.items()},
            "composite":   composite_table,
            "t1_islands":  t1_islands,
            "t2_islands":  t2_islands,
            "events":      self._events[-30:],
        }
        try:
            payload = json.dumps(state, cls=NumpyEncoder)
            tmp = STATE_FILE.with_suffix(".tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(STATE_FILE)   # atomic on POSIX
        except Exception as e:
            print(f"[controller] STATE DUMP ERROR: {e}", flush=True)

    # ── Stagnation helpers ────────────────────────────────────────────────────
    def _inject_random(self, opt, tag, n_inject):
        fresh = self._eval_batch(
            [np.random.rand(self.dims) for _ in range(n_inject)], tag)
        if hasattr(opt, "population") and opt.population:
            opt.population.sort(key=lambda x: x.fitness, reverse=True)
            opt.population = opt.population[:-n_inject] + fresh
        return opt

    # ── Main run ─────────────────────────────────────────────────────────────
    def run(self):
        # ── PHASE 1: INITIALIZATION ───────────────────────────────────────
        self._phase = "INIT"
        print(_sep())
        print(_hdr("PHASE 1 — INITIALIZATION  (N = 50)"))
        print(_sep())

        init_pop = []
        while len(init_pop) < 50 and self.evals_used < self.budget:
            init_pop.extend(
                self._eval_batch([np.random.rand(self.dims)], "RandomInit"))
        random.shuffle(init_pop)   # random (NOT fitness-sorted) assignment

        # 40% → T1 (20 inds), 60% → T2 pool (30 inds, unassigned)
        t1_pool = init_pop[:20]
        t2_pool = list(init_pop[20:50])

        print(f"  Generated {len(init_pop)} individuals | "
              f"T1 pool: {len(t1_pool)} | T2 pool (unassigned): {len(t2_pool)}")
        fitnesses = sorted([i.fitness for i in init_pop], reverse=True)
        print(f"  Best: {fitnesses[0]:.2f}%  Avg: {sum(fitnesses)/len(fitnesses):.2f}%")

        # ── PHASE 2: T1 WARM-UP (5 iters, 4 algos × 5 inds each) ─────────
        self._phase = "T1_WARMUP"
        print(f"\n{_sep()}")
        print(_hdr("PHASE 2 — TIER-1 WARM-UP  (5 iters × 4 algorithms)"))
        print(_sep())

        # Split 20 T1 individuals among 4 algorithms (5 each)
        t1_optims: dict[str, object] = {}
        for i, alg in enumerate(self.ALL_ALGOS):
            seed = t1_pool[i*5:(i+1)*5]
            opt  = _make_opt(alg, self.space, 5)
            opt.tell(seed)
            t1_optims[alg] = opt
            best = max(ind.fitness for ind in seed)
            print(f"  T1-{alg:5s}: seeded {len(seed)} inds | best {best:.2f}%")

        print(f"\n  Running {5} warm-up iterations...")
        for wu in range(1, 6):
            if _STOP_FLAG.is_set():
                break
            time.sleep(self.step_delay)
            self._iteration = wu
            for alg, opt in t1_optims.items():
                if self.evals_used >= self.budget:
                    break
                inds = self._eval_batch(opt.ask(5), f"T1_{alg}")
                if inds:
                    opt.tell(inds)
            self._dump_state(t1_optims, {}, {f"T1_{k}": 0.0 for k in t1_optims}, {}, {})

        # ── PHASE 3: COMPOSITE SCORING + INITIAL PROMOTION ────────────────
        self._phase = "SCORING"
        print(f"\n{_sep()}")
        print(_hdr("PHASE 3 — COMPOSITE SCORING + INITIAL T2 PROMOTION"))
        print(_sep())

        t1_named = {f"T1_{k}": list(v.population)
                    for k, v in t1_optims.items()}
        t1_sc, t1_raw = self._composite(t1_named)

        print(f"\n  {'Algorithm':<10} {'Accuracy':>8} {'Speed':>8} "
              f"{'Diversity':>10} {'Composite':>10}")
        print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
        for lbl in sorted(t1_sc, key=t1_sc.get, reverse=True):
            r = t1_raw[lbl]
            print(f"  {lbl:<10} {r['fitness']:>7.2f}% {r['speed']:>8.3f} "
                  f"{r['diversity']:>10.3f} {t1_sc[lbl]:>10.3f}")

        sorted_algos = sorted(t1_sc, key=t1_sc.get, reverse=True)
        t2_names  = [s.replace("T1_", "") for s in sorted_algos[:2]]
        t1_remain = [s.replace("T1_", "") for s in sorted_algos[2:]]
        print(f"\n  PROMOTED to T2 : {t2_names}   (ranked #1, #2)")
        print(f"  REMAIN in T1   : {t1_remain}  (ranked #3, #4)")

        for alg in self.ALL_ALGOS:
            self.bandit.update(alg, t1_sc.get(f"T1_{alg}", 0.0))

        # Build T2 islands — each gets 15 individuals
        # (5 from their T1 batch + 10 from the T2 pool)
        t2_optims: dict[str, object] = {}
        for alg in t2_names:
            seed = list(t1_optims[alg].population)          # 5 from T1
            extra_needed = 15 - len(seed)
            extra = t2_pool[:extra_needed]
            t2_pool = t2_pool[extra_needed:]
            combined = seed + extra
            opt = _make_opt(alg, self.space, 15)
            opt.tell(combined)
            t2_optims[alg] = opt
            best = max(i.fitness for i in combined)
            print(f"  T2-{alg:5s}: {len(combined)} inds | best {best:.2f}%")

        # T1 keeps only the two remaining algorithms
        t1_optims = {a: t1_optims[a] for a in t1_remain}
        t1_stag   = {a: 0 for a in t1_optims}
        t2_stag   = {a: 0 for a in t2_optims}

        # Running composite tables (empty until first scoring)
        all_t1_sc, all_t2_sc, all_raw = {}, {}, {}
        self._dump_state(t1_optims, t2_optims,
                         {f"T1_{k}": 0.0 for k in t1_optims},
                         {f"T2_{k}": 0.0 for k in t2_optims}, {})

        # ── PHASE 5: PARALLEL MAIN LOOP ───────────────────────────────────
        self._phase = "PARALLEL"
        print(f"\n{_sep()}")
        print(_hdr(f"PHASE 5 — PARALLEL MAIN LOOP  [T2: {list(t2_optims)} | T1: {list(t1_optims)}]"))
        print(_sep())
        print(f"  k={self.k} | elite_pct={int(self.elite_pct*100)}% | "
              f"inject={int(self.inject_pct*100)}% | threshold={self.swap_thr}")
        print(f"\n  {'Iter':>4}  {'T2 Best':>20}  {'T1 Best':>20}  "
              f"{'Global':>8}  {'Evals':>8}")
        print(f"  {'-'*4}  {'-'*20}  {'-'*20}  {'-'*8}  {'-'*8}")

        iteration        = 5
        top_down_counter = 0   # counts iters for every-10-iter T2→T1 injection
        t1_stag_rounds   = {a: 0 for a in t1_optims}
        t2_stag_rounds   = {a: 0 for a in t2_optims}
        global_stag_cnt  = 0

        while self.evals_used < self.budget:
            # Check stop flag (set by /api/stop endpoint)
            if _STOP_FLAG.is_set():
                self._event("Run stopped by user.", "INFO")
                break
            time.sleep(self.step_delay)   # pace so UI can see iterations
            iteration       += 1
            self._iteration  = iteration
            top_down_counter += 1
            prev_best = self.best_individual.fitness if self.best_individual else 0.0

            # ── Evolve all T2 islands ────────────────────────────────────
            for alg, opt in t2_optims.items():
                if self.evals_used >= self.budget: break
                inds = self._eval_batch(opt.ask(15), f"T2_{alg}")
                if inds: opt.tell(inds)

            # ── Evolve all T1 islands ────────────────────────────────────
            for alg, opt in t1_optims.items():
                if self.evals_used >= self.budget: break
                inds = self._eval_batch(opt.ask(5), f"T1_{alg}")
                if inds: opt.tell(inds)

            if self.evals_used >= self.budget:
                break

            # ── Print per-iteration row ──────────────────────────────────
            glob  = self.best_individual.fitness if self.best_individual else 0.0
            t2_str = "  ".join(f"{a}:{max((i.fitness for i in o.population),default=0):.1f}%"
                                for a, o in t2_optims.items())
            t1_str = "  ".join(f"{a}:{max((i.fitness for i in o.population),default=0):.1f}%"
                                for a, o in t1_optims.items())
            flag = " *** NEW BEST ***" if glob > prev_best else ""
            print(f"  {iteration:>4}  {t2_str:>20}  {t1_str:>20}  "
                  f"{glob:>7.2f}%  {self.evals_used:>4}/{self.budget}{flag}")

            # ── Every k iterations: switching strategies ─────────────────
            if iteration % self.k == 0:
                print(f"\n  {_sep('-',60)}")
                print(f"  {_hdr('SWITCHING STRATEGY CHECKPOINT', '-', 60)}")

                # Build named pops for composite scoring
                named = {}
                named.update({f"T2_{k}": list(v.population) for k, v in t2_optims.items()})
                named.update({f"T1_{k}": list(v.population) for k, v in t1_optims.items()})
                sc, raw = self._composite(named)
                all_raw = raw

                t2_sc_now = {k: sc.get(f"T2_{k}", 0.0) for k in t2_optims}
                t1_sc_now = {k: sc.get(f"T1_{k}", 0.0) for k in t1_optims}
                all_t1_sc = {f"T1_{k}": v for k, v in t1_sc_now.items()}
                all_t2_sc = {f"T2_{k}": v for k, v in t2_sc_now.items()}

                # Print composite table
                print(f"  {'Label':<12} {'Accuracy':>8} {'Diversity':>10} {'Composite':>10}")
                print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10}")
                for lbl in sorted(sc, key=sc.get, reverse=True):
                    r = raw.get(lbl, {})
                    print(f"  {lbl:<12} {r.get('fitness',0):>7.2f}%"
                          f" {r.get('diversity',0):>10.3f}"
                          f" {sc.get(lbl,0):>10.3f}")

                # Update UCB1
                for k in t1_optims:
                    self.bandit.update(k, t1_sc_now.get(k, 0.0))
                for k in t2_optims:
                    self.bandit.update(k, t2_sc_now.get(k, 0.0))

                # ── a) ELITE SHARING between T2 islands ─────────────────
                t2_keys = list(t2_optims.keys())
                if len(t2_keys) >= 2:
                    a1, a2 = t2_keys[0], t2_keys[1]
                    p1 = sorted(t2_optims[a1].population,
                                key=lambda x: x.fitness, reverse=True)
                    p2 = sorted(t2_optims[a2].population,
                                key=lambda x: x.fitness, reverse=True)
                    n_share = max(1, int(min(len(p1), len(p2)) * self.elite_pct))
                    t2_optims[a1].population = p1[:-n_share] + p2[:n_share]
                    t2_optims[a2].population = p2[:-n_share] + p1[:n_share]
                    self._event(f"ELITE SHARE: top {n_share} exchanged "
                                f"between T2-{a1} ↔ T2-{a2}", "SHARE")

                # ── b) CHALLENGE CHECK (T1 beats a T2 island) ──────────
                action_taken = False
                if t1_sc_now and t2_sc_now:
                    best_t1_key  = max(t1_sc_now, key=t1_sc_now.get)
                    worst_t2_key = min(t2_sc_now, key=t2_sc_now.get)
                    t1_best_sc   = t1_sc_now[best_t1_key]
                    t2_worst_sc  = t2_sc_now[worst_t2_key]

                    if t1_best_sc > t2_worst_sc + self.swap_thr:
                        self._event(
                            f"PROMOTION: T1-{best_t1_key}({t1_best_sc:.3f}) "
                            f"> T2-{worst_t2_key}({t2_worst_sc:.3f}) → SWAP!",
                            "PROMOTE")
                        old_t2_pop = list(t2_optims[worst_t2_key].population)
                        old_t1_pop = list(t1_optims[best_t1_key].population)

                        new_t2 = _make_opt(best_t1_key, self.space, 15)
                        new_t2.tell(old_t1_pop + old_t2_pop[:15-len(old_t1_pop)])
                        new_t1 = _make_opt(worst_t2_key, self.space, 5)
                        new_t1.tell(old_t2_pop[:5])

                        del t2_optims[worst_t2_key]
                        del t1_optims[best_t1_key]
                        t2_optims[best_t1_key]  = new_t2
                        t1_optims[worst_t2_key] = new_t1
                        t2_stag_rounds[best_t1_key]  = 0
                        t1_stag_rounds[worst_t2_key] = 0
                        action_taken = True

                # ── c) STAGNATION — T2 ─────────────────────────────────
                if not action_taken:
                    for alg in list(t2_optims.keys()):
                        r = raw.get(f"T2_{alg}", {})
                        if r.get("trend", 1.0) <= 1e-4:
                            t2_stag_rounds[alg] = t2_stag_rounds.get(alg,0)+1
                            self._event(
                                f"STAGNATION T2-{alg}: "
                                f"{t2_stag_rounds[alg]}/{self.stag_limit} checks",
                                "STAG")
                            if t2_stag_rounds[alg] >= self.stag_limit:
                                n_inj = max(1, int(15 * self.inject_pct))
                                t2_optims[alg] = self._inject_random(
                                    t2_optims[alg], f"T2_{alg}", n_inj)
                                self._event(
                                    f"INJECT {n_inj} random inds into T2-{alg}", "INJECT")
                                # Persistent stagnation → UCB1 bandit swap
                                if t2_stag_rounds[alg] > self.stag_limit:
                                    candidates = [a for a in self.ALL_ALGOS
                                                  if a not in t2_optims]
                                    if candidates:
                                        chosen = self.bandit.select_next(candidates)
                                        self._event(
                                            f"UCB1 SWAP: replace T2-{alg} "
                                            f"with {chosen}", "UCB1")
                                        old_pop = list(t2_optims[alg].population)
                                        new_t2  = _make_opt(chosen, self.space, 15)
                                        new_t2.tell(old_pop)
                                        del t2_optims[alg]
                                        t2_optims[chosen] = new_t2
                                        t2_stag_rounds[chosen] = 0
                                        break
                        else:
                            t2_stag_rounds[alg] = 0

                # ── d) STAGNATION — T1 single strategy ─────────────────
                all_t1_flat = True
                for alg in list(t1_optims.keys()):
                    r = raw.get(f"T1_{alg}", {})
                    if r.get("trend", 1.0) <= 1e-4:
                        t1_stag_rounds[alg] = t1_stag_rounds.get(alg, 0) + 1
                        if t1_stag_rounds[alg] >= self.stag_limit:
                            # Refresh bottom 50% of T1 batch
                            pop = sorted(t1_optims[alg].population,
                                         key=lambda x: x.fitness, reverse=True)
                            n_refresh = max(1, len(pop)//2)
                            fresh = self._eval_batch(
                                [np.random.rand(self.dims) for _ in range(n_refresh)],
                                f"T1_{alg}")
                            t1_optims[alg].population = pop[:-n_refresh] + fresh
                            self._event(
                                f"T1-{alg} REFRESH: replaced bottom {n_refresh} inds",
                                "STAG")
                            t1_stag_rounds[alg] = 0
                    else:
                        t1_stag_rounds[alg] = 0
                        all_t1_flat = False

                # ── e) STAGNATION — ALL T1 strategies flat ─────────────
                if all_t1_flat and len(t1_optims) >= 2:
                    global_stag_cnt += 1
                    if global_stag_cnt >= self.stag_limit:
                        self._event("GLOBAL T1 STAGNATION: keeping 1 best "
                                    "per island, refreshing rest + 3-iter re-warmup",
                                    "GLOBAL_STAG")
                        for alg, opt in t1_optims.items():
                            pop = sorted(opt.population,
                                         key=lambda x: x.fitness, reverse=True)
                            keeper = [pop[0]]
                            fresh  = self._eval_batch(
                                [np.random.rand(self.dims) for _ in range(4)],
                                f"T1_{alg}")
                            opt.population = keeper + fresh
                        # 3-iteration mini warm-up
                        for _ in range(3):
                            for alg, opt in t1_optims.items():
                                inds = self._eval_batch(opt.ask(5), f"T1_{alg}")
                                if inds: opt.tell(inds)
                        global_stag_cnt = 0
                else:
                    global_stag_cnt = 0

                print(f"  {_sep('-',60)}\n")
                print(f"  {'Iter':>4}  {'T2 Best':>20}  {'T1 Best':>20}  "
                      f"{'Global':>8}  {'Evals':>8}")
                print(f"  {'-'*4}  {'-'*20}  {'-'*20}  {'-'*8}  {'-'*8}")

            # ── PHASE 8: TOP-DOWN T2→T1 INJECTION every 10 iters ────────
            if top_down_counter >= 10:
                top_down_counter = 0
                for t2_alg, t2_opt in t2_optims.items():
                    if not t2_opt.population: continue
                    best_ind = max(t2_opt.population, key=lambda x: x.fitness)
                    for t1_alg, t1_opt in t1_optims.items():
                        if not t1_opt.population: continue
                        worst_idx = min(range(len(t1_opt.population)),
                                        key=lambda j: t1_opt.population[j].fitness)
                        if best_ind.fitness > t1_opt.population[worst_idx].fitness:
                            t1_opt.population[worst_idx] = best_ind
                self._event(
                    f"TOP-DOWN SHARE: best T2 individuals injected into all T1 islands",
                    "TOPDOWN")

            # ── Write state.json after every iteration ───────────────────
            self._dump_state(
                t1_optims, t2_optims,
                all_t1_sc, all_t2_sc, all_raw)

        # ── CONVERGENCE ───────────────────────────────────────────────────
        self._phase = "CONVERGED"
        print(f"\n{_sep()}")
        print(_hdr("PHASE 9 — CONVERGENCE / FINAL RESULTS"))
        print(_sep())

        # Ensemble: top-5 individuals across all evaluated configs
        top5 = sorted(self.all_individuals, key=lambda x: x.fitness, reverse=True)[:5]

        print(f"  Total Evaluations : {self.evals_used}")
        print(f"  Elapsed           : {round(time.time()-self._start_time,1)}s")
        print(f"  Winning Algorithm : {self.best_individual.source_algorithm}")
        print(f"  Best Accuracy     : {self.best_individual.fitness:.4f}%")
        print(f"\n  Best Hyperparameters:")
        params = _decode(self.best_individual, self.space)
        for k, v in params.items():
            val = f"{v:.6f}" if isinstance(v, float) else str(v)
            print(f"    * {k:<20}: {val}")

        print(f"\n  Top-5 Ensemble:")
        for rank, ind in enumerate(top5, 1):
            print(f"    {rank}. {ind.source_algorithm:<12} {ind.fitness:.4f}%")

        # Final state dump
        self._dump_state(t1_optims, t2_optims, all_t1_sc, all_t2_sc, all_raw)
        return self.best_individual
