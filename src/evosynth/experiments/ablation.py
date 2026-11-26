"""
Ablation: isolated GA / PSO / DE / CMA-ES vs the EvoSynth meta-optimizer. Command: `evosynth ablation`.
Settings: [experiments.ablation] (setups, pop_size, batch sizes, EvoSynth t1/k schedule,
evosynth_splits, passive_marks), [composite], [bandit], [datasets.<name>].

Every method gets the same budget, population, evaluator (search rows, CV folds, metric) and seed.

  Isolated optimizers   run one after another in strict isolation (no tiers, no sharing, no
                        injection). Every evaluation is checkpointed; an interrupted run resumes
                        with the optimizer's full internal state (PSO particles, DE targets, CMA-ES
                        distribution) and random generators (Phase 4.6).
  Passive scoring       composite scores are relative (min-max over the families scored together),
  (Phase 4.2)           so the four isolated runs are scored TOGETHER at the same evaluation marks
                        with the controller's own CompositeScoringEngine and UCB1 bonus: what the
                        bandit would have measured had it watched them. Nothing is decided from it.
  EvoSynth (Phase 4.1)  diversity / composite / UCB come from the controller's own scoring history
                        (DynamicOptimizer.scoring_history), never from constants. One EvoSynth run per
                        tier split in evosynth_splits / --splits (split ablation, Phase 1.6).
  Conclusion (4.3)      generated from the numbers, whichever method wins.

Output: results/ablation/<dataset>/<setup>/seed_<k>/: run.log, events.jsonl, manifest.json,
result.json, summary.txt, individual_runs/<setup>_<method>.txt, isolated_population.csv,
population[_<split>].csv (EvoSynth), checkpoints/ (resume state).
"""
import concurrent.futures
import json
import os
import pickle
import time

import numpy as np

from evosynth.config import get_settings
from evosynth.core.decoding import decode_vector_to_dict
from evosynth.core.individual import Individual
from evosynth.engine.bandit import UCB1Bandit
from evosynth.engine.controller import DynamicOptimizer
from evosynth.engine.scoring import CompositeScoringEngine
from evosynth.evaluators.tabular import TabularEvaluator
from evosynth.experiments.checkpointing import atomic_json_dump, atomic_pickle_dump
from evosynth.experiments.run_context import RunContext
from evosynth.optimizers.cmaes import CMAESOptimizer
from evosynth.optimizers.de import DifferentialEvolution
from evosynth.optimizers.ga import GeneticAlgorithm
from evosynth.optimizers.pso import ParticleSwarm
from evosynth.search_spaces.tabular import get_tabular_search_space
from evosynth.seeding import stream

ALGOS = ("GA", "PSO", "DE", "CMAES")
NAMES = {"GA": "Genetic Algorithm", "PSO": "Particle Swarm Optimization", "DE": "Differential Evolution",
         "CMAES": "CMA-ES"}
_CLASSES = {"GA": GeneticAlgorithm, "PSO": ParticleSwarm, "DE": DifferentialEvolution, "CMAES": CMAESOptimizer}


def _diversity(pop, dims):
    return DynamicOptimizer._latent_diversity(pop, dims)


# ------------------------------------------------------------------------------ isolated runs
def run_isolated_optimizer(algo_key, space, evaluator, budget, pop_size, max_workers, ckpt_dir,
                           checkpoint_interval=10):
    """One optimizer in strict isolation. Returns a dict with its log, individuals and best."""
    dims = len(list(space.keys()))
    state_file = os.path.join(ckpt_dir, f"{algo_key}_state.pkl")
    ab = get_settings().experiments.ablation
    batch = max(ab.batch_min, min(ab.batch_max, pop_size // ab.batch_divisor))

    st = None
    if os.path.exists(state_file):
        try:
            with open(state_file, "rb") as f:
                st = pickle.load(f)
            print(f"  [RESUME] {algo_key}: continuing from evaluation {st['evals']}/{budget}"
                  f"{' (completed)' if st['evals'] >= budget else ''}")
        except Exception as e:
            print(f"  [WARNING] Could not restore {algo_key} ({e}); starting fresh.")
            st = None
    if st is None:
        st = {"evals": 0, "inds": [], "log": [], "eval_seconds": 0.0, "wall_seconds": 0.0,
              "warmup_rng": stream("ablation", algo_key, "warmup"),
              "optimizer": _CLASSES[algo_key](space, pop_size, rng=stream("ablation", algo_key, "optimizer")),
              "warmed_up": False}
    optimizer, rng = st["optimizer"], st["warmup_rng"]

    def _save():
        atomic_pickle_dump(state_file, st)

    def _eval_single(vec):
        ind = Individual(vec)
        ind.set_fitness(evaluator.evaluate(ind))
        ind.source_algorithm = f"Isolated_{algo_key}"
        return ind

    def _evaluate(vectors):
        vectors = vectors[:budget - st["evals"]]
        if not vectors:
            return []
        t = time.perf_counter()
        workers = min(max_workers, len(vectors))
        if workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                inds = list(ex.map(_eval_single, vectors))
        else:
            inds = [_eval_single(v) for v in vectors]
        st["eval_seconds"] += time.perf_counter() - t
        st["evals"] += len(inds)
        st["inds"].extend(inds)
        return inds

    def _log(flag=""):
        best = max(i.fitness for i in st["inds"])
        st["log"].append({"evals": st["evals"], "best": best, "diversity": _diversity(optimizer.population, dims),
                          "eval_seconds": st["eval_seconds"], "flag": flag})

    start = time.time()
    if not st["warmed_up"]:                                   # random initial population
        init = []
        while len(init) < pop_size and st["evals"] < budget:
            init += _evaluate([rng.random(dims) for _ in range(min(batch, pop_size - len(init)))])
        optimizer.tell(init)
        st["warmed_up"] = True
        _log("warm-up")
        _save()
    last_saved = st["evals"]
    while st["evals"] < budget:
        prev = max(i.fitness for i in st["inds"])
        inds = _evaluate(optimizer.ask(min(batch, budget - st["evals"])))
        if inds:
            optimizer.tell(inds)
        _log("NEW BEST" if max(i.fitness for i in inds) > prev else "")
        if st["evals"] - last_saved >= checkpoint_interval or st["evals"] >= budget:
            st["wall_seconds"] += time.time() - start
            start = time.time()
            _save()
            last_saved = st["evals"]
    st["wall_seconds"] += time.time() - start
    _save()

    best = max(st["inds"], key=lambda i: i.fitness)
    return {"algo_key": algo_key, "best_ind": best, "best_score": best.fitness, "total_evals": st["evals"],
            "total_seconds": st["wall_seconds"], "log": st["log"], "inds": st["inds"],
            "params": decode_vector_to_dict(best.vector, space)}


def _at(log, mark):
    """The last log entry at or before `mark` evaluations."""
    rows = [r for r in log if r["evals"] <= mark]
    return rows[-1] if rows else log[0]


def passive_scores(isolated, marks):
    """
    Score the isolated runs together at each mark with the controller's formulas (Phase 4.2):
    fitness = best so far; speed = improvement since the previous mark; efficiency = that
    improvement per evaluation-second; diversity = latent diversity of the population; trend =
    slope of the best over the last trend_window marks; composite = CompositeScoringEngine over
    the four; ucb = composite + UCB1 bonus (equal for all: every run has used `mark` evaluations).
    Returns {mark: {algo: {metric: value}}}.
    """
    s = get_settings()
    engine, bandit, window = CompositeScoringEngine(), UCB1Bandit(), s.composite.trend_window
    out, history = {}, {a: [] for a in isolated}
    prev = {a: {"best": 0.0, "eval_seconds": 0.0} for a in isolated}
    for m in marks:
        raw = {}
        for a, res in isolated.items():
            row = _at(res["log"], m)
            history[a].append(row["best"])
            h = history[a][-window:]
            improvement = max(row["best"] - prev[a]["best"], 0.0)
            seconds = max(row["eval_seconds"] - prev[a]["eval_seconds"], 1e-6)
            raw[a] = {"fitness": row["best"], "speed": improvement, "efficiency": improvement / seconds,
                      "diversity": row["diversity"],
                      "trend": float(np.polyfit(np.arange(len(h)), h, 1)[0]) if len(h) >= 2 else 0.0}
            prev[a] = row
        comp = engine.calculate_scores(raw)
        bonus = bandit.bonus(m, m * len(isolated))
        out[m] = {a: {**raw[a], "composite": comp[a], "ucb": comp[a] + bonus} for a in raw}
    return out


# ------------------------------------------------------------------------------ EvoSynth
def _evosynth_at(opt, mark):
    """EvoSynth's own measurements at the last scoring call at or before `mark` (Phase 4.1)."""
    calls = [c for c in opt["scoring_history"] if c["evals"] <= mark] or opt["scoring_history"][:1]
    best = max((h["best_so_far"] for h in opt["history"] if h["eval_id"] <= mark), default=0.0)
    if not calls:
        return {"best": best, "diversity": None, "composite": None, "ucb": None, "top_island": None}
    islands = calls[-1]["islands"]
    top = max(islands, key=lambda n: islands[n]["ucb"])
    return {"best": best, "diversity": float(np.mean([v["diversity"] for v in islands.values()])),
            "composite": islands[top]["composite"], "ucb": islands[top]["ucb"], "top_island": top,
            "scoring_call_evals": calls[-1]["evals"]}


def run_evosynth(run, split, space, evaluator, budget, pop_size, max_workers, ckpt_dir):
    """One EvoSynth run with tier split `split`; resumes only as a whole (it is deterministic)."""
    tag = split if split != get_settings().engine.tier_split else ""
    result_file = os.path.join(ckpt_dir, f"EvoSynth{('_' + tag) if tag else ''}.pkl")
    if os.path.exists(result_file):
        with open(result_file, "rb") as f:
            print(f"  [RESUME] EvoSynth ({split} split) already completed.")
            return pickle.load(f)
    ab = get_settings().experiments.ablation
    t0 = time.time()
    opt = DynamicOptimizer(search_space=space, evaluator=evaluator, budget=budget, pop_size=pop_size,
                           t1_iters=max(ab.t1_iters_min, int(budget * ab.t1_iters_fraction)),
                           k_iters=max(ab.k_iters_min, int(budget * ab.k_iters_fraction)),
                           max_workers=max_workers,
                           population_csv=run.file(f"population{('_' + tag) if tag else ''}.csv"))
    opt.tier_split = split
    best = opt.run()
    out = {"split": split, "best_ind": best, "best_score": best.fitness, "total_evals": opt.evals_used,
           "total_seconds": time.time() - t0, "params": decode_vector_to_dict(best.vector, space),
           "winning_algorithm": best.source_algorithm, "scoring_history": opt.scoring_history,
           "decisions": opt.decisions, "history": opt.history}
    atomic_pickle_dump(result_file, out)
    return out


# ------------------------------------------------------------------------------ report
def _conclusion(evo, best_iso, iso_name, cp_evo, cp_iso, checkpoint_eval, budget, metric):
    d_final, d_cp = evo - best_iso, cp_evo - cp_iso

    def says(d):
        if abs(d) < 0.005:
            return "matched"
        return f"{'beat' if d > 0 else 'trailed'} by {abs(d):.2f} points"
    return [
        f"At the full budget ({budget} evaluations) EvoSynth ({evo:.2f}%) {says(d_final)} the best isolated "
        f"optimizer, {iso_name} ({best_iso:.2f}%), on search CV {metric}.",
        f"At the {checkpoint_eval}-evaluation mark EvoSynth ({cp_evo:.2f}%) {says(d_cp)} the best isolated "
        f"optimizer at that mark ({cp_iso:.2f}%).",
        "This is one seed. Compare the mean and spread over seeds with `evosynth summary` before drawing conclusions.",
    ]


def _fmt(v, spec=".3f"):
    return "-" if v is None else format(v, spec)


def run_benchmark_setup(run, setup_name, budget, pop_size, evaluator, space, max_workers, checkpoint_eval,
                        checkpoint_interval=10, splits=None):
    s = get_settings()
    ab = s.experiments.ablation
    splits = list(splits or ab.evosynth_splits)
    ckpt_dir, ind_dir = run.file("checkpoints"), run.file("individual_runs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(ind_dir, exist_ok=True)
    state_file = os.path.join(ckpt_dir, "experiment_state.json")
    state = {}
    if os.path.exists(state_file):
        with open(state_file, encoding="utf-8") as f:
            state = json.load(f)
    metric = evaluator.metric

    print("=" * 80)
    print(f" EVOSYNTH ABLATION — {setup_name} on {evaluator.dataset} ({metric})")
    print(f" Budget {budget} | population {pop_size} | mark @ {checkpoint_eval} evals | EvoSynth splits: {splits}")
    print("=" * 80)

    isolated = {}
    for a in ALGOS:
        print(f"\n[RUNNING] Isolated {a} (budget {budget}, population {pop_size})")
        state[a] = "running"; atomic_json_dump(state_file, state)
        isolated[a] = run_isolated_optimizer(a, space, evaluator, budget, pop_size, max_workers, ckpt_dir,
                                             checkpoint_interval)
        state[a] = "completed"; atomic_json_dump(state_file, state)
        print(f"  [OK] {a}: best {isolated[a]['best_score']:.2f}% ({isolated[a]['params'].get('model_type')})")

    evos = {}
    for split in splits:
        name = "EvoSynth" if split == s.engine.tier_split else f"EvoSynth[{split}]"
        print(f"\n[RUNNING] {name} (tier split: {split})")
        state[name] = "running"; atomic_json_dump(state_file, state)
        evos[name] = run_evosynth(run, split, space, evaluator, budget, pop_size, max_workers, ckpt_dir)
        state[name] = "completed"; atomic_json_dump(state_file, state)

    marks = sorted({int(budget * i / ab.passive_marks) for i in range(1, ab.passive_marks + 1)} | {checkpoint_eval})
    passive = passive_scores(isolated, marks)

    # ---------------------------------------------------------------- per-method entries
    methods = {}
    for a, res in isolated.items():
        cp, fin = passive[checkpoint_eval][a], passive[budget][a]
        methods[a] = {"best_cv_score": res["best_score"], "score_at_mark": cp["fitness"],
                      "diversity_at_mark": cp["diversity"], "composite_at_mark": cp["composite"], "ucb_at_mark": cp["ucb"],
                      "final_diversity": fin["diversity"], "final_composite": fin["composite"], "final_ucb": fin["ucb"],
                      "total_evals": res["total_evals"], "total_seconds": round(res["total_seconds"], 2),
                      "best_config": res["params"],
                      "curve": [[m, passive[m][a]["fitness"]] for m in marks]}
    for name, res in evos.items():
        cp, fin = _evosynth_at(res, checkpoint_eval), _evosynth_at(res, budget)
        kinds = [d["action"] for d in res["decisions"]]
        methods[name] = {"best_cv_score": res["best_score"], "score_at_mark": cp["best"],
                         "diversity_at_mark": cp["diversity"], "composite_at_mark": cp["composite"],
                         "ucb_at_mark": cp["ucb"], "final_diversity": fin["diversity"],
                         "final_composite": fin["composite"], "final_ucb": fin["ucb"],
                         "top_island_final": fin["top_island"], "tier_split": res["split"],
                         "winning_algorithm": res["winning_algorithm"],
                         "decisions": {k: kinds.count(k) for k in sorted(set(kinds))},
                         "total_evals": res["total_evals"], "total_seconds": round(res["total_seconds"], 2),
                         "best_config": res["params"],
                         "curve": [[m, _evosynth_at(res, m)["best"]] for m in marks]}

    best_iso = max(ALGOS, key=lambda a: methods[a]["best_cv_score"])
    best_iso_cp = max(ALGOS, key=lambda a: methods[a]["score_at_mark"])
    main = "EvoSynth" if "EvoSynth" in methods else next(iter(evos))
    conclusion = _conclusion(methods[main]["best_cv_score"], methods[best_iso]["best_cv_score"], best_iso,
                             methods[main]["score_at_mark"], methods[best_iso_cp]["score_at_mark"],
                             checkpoint_eval, budget, metric)

    # ---------------------------------------------------------------- files
    for name, m in methods.items():
        with open(os.path.join(ind_dir, f"{setup_name}_{name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"{setup_name} — {name} on {evaluator.dataset} (search CV {metric}, budget {budget}, "
                    f"population {pop_size})\n\n")
            if name in isolated:
                f.write(f"  {'Evals':>6} {'Best':>8} {'Diversity':>10} {'Flag':<10}\n")
                for r in isolated[name]["log"]:
                    f.write(f"  {r['evals']:>6} {r['best']:>7.2f}% {r['diversity']:>10.3f} {r['flag']:<10}\n")
            else:
                f.write(f"  Tier split: {m['tier_split']} | decisions: {m['decisions']}\n")
                f.write(f"  {'Evals':>6} {'Island':<10} {'Best':>8} {'Diversity':>10} {'Composite':>10} {'UCB':>7}\n")
                for c in evos[name]["scoring_history"]:
                    for isl, v in c["islands"].items():
                        f.write(f"  {c['evals']:>6} {isl:<10} {v['fitness']:>7.2f}% {v['diversity']:>10.3f} "
                                f"{v['composite']:>10.3f} {v['ucb']:>7.3f}\n")
            f.write(f"\n  Best: {m['best_cv_score']:.2f}%\n  Configuration: {json.dumps(m['best_config'], default=str)}\n")

    with open(run.file("isolated_population.csv"), "w", encoding="utf-8") as f:
        keys = list(space.keys())
        f.write("algorithm,eval_id,fitness," + ",".join(keys) + "\n")
        for a, res in isolated.items():
            for i, ind in enumerate(res["inds"], 1):
                p = decode_vector_to_dict(ind.vector, space)
                f.write(f"{a},{i},{ind.fitness:.4f}," + ",".join(str(p.get(k, "")) for k in keys) + "\n")

    with open(run.file("summary.txt"), "w", encoding="utf-8") as f:
        w = 118
        f.write("=" * w + f"\n ABLATION {setup_name}: isolated optimizers vs EvoSynth — {evaluator.dataset}, "
                f"search CV {metric}, budget {budget}, population {pop_size}\n" + "=" * w + "\n")
        f.write(" Isolated runs: composite / UCB are passive (the four scored together with the controller's\n"
                " formulas, nothing decided). EvoSynth: values of its highest-UCB island at its last scoring call;\n"
                " diversity = mean over its islands.\n\n")
        for title, key_s, key_d, key_c, key_u, at in (
                (f"AT {checkpoint_eval} EVALUATIONS", "score_at_mark", "diversity_at_mark", "composite_at_mark", "ucb_at_mark", checkpoint_eval),
                (f"AT {budget} EVALUATIONS (FULL BUDGET)", "best_cv_score", "final_diversity", "final_composite", "final_ucb", budget)):
            f.write("-" * w + f"\n {title}\n" + "-" * w + "\n")
            f.write(f" {'Method':<26} {'Best':>8} {'Diversity':>10} {'Composite':>10} {'UCB':>7}  {'Model':<13} {'FE':<13}\n")
            for name, m in methods.items():
                f.write(f" {name:<26} {m[key_s]:>7.2f}% {_fmt(m[key_d]):>10} {_fmt(m[key_c]):>10} {_fmt(m[key_u]):>7}  "
                        f"{m['best_config'].get('model_type', ''):<13} {m['best_config'].get('fe_strategy', ''):<13}\n")
            f.write("\n")
        for name in evos:
            f.write(f" {name} decisions: {methods[name]['decisions']}; best found by {methods[name]['winning_algorithm']}\n")
        f.write("\n CONCLUSION (generated from the numbers above)\n")
        for i, line in enumerate(conclusion, 1):
            f.write(f"   {i}. {line}\n")
        f.write("=" * w + "\n")

    run.write_result({"setup": setup_name, "dataset": evaluator.dataset, "budget": budget, "pop_size": pop_size,
                      "metric": metric, "checkpoint_eval": checkpoint_eval, "marks": marks,
                      "best_isolated": best_iso,
                      "delta_vs_best_isolated": methods[main]["best_cv_score"] - methods[best_iso]["best_cv_score"],
                      "conclusion": conclusion, "methods": methods})
    print("\n".join(["", " CONCLUSION"] + [f"   {i}. {c}" for i, c in enumerate(conclusion, 1)]))
    print(f"\n[ALL DONE] Results in {run.path}")
    return methods


def run_ablation(setup=None, pop_size=None, max_workers=None, checkpoint_interval=None,
                 results_dir=None, dataset="covertype", splits=None):
    """Isolated GA/PSO/DE/CMA-ES vs EvoSynth for one setup from [experiments.ablation].setups."""
    ab = get_settings().experiments.ablation
    setups = dict(ab.setups)
    setup = setup or next(iter(setups))
    if setup not in setups:
        raise ValueError(f"unknown setup '{setup}'. Available: {sorted(setups)}")
    budget = setups[setup]
    space = get_tabular_search_space(include_model_type=True, include_fe_strategy=True)
    evaluator = TabularEvaluator(dataset)
    # resume=True: an interrupted setup continues from checkpoints/ and appends to run.log
    with RunContext("ablation", dataset, setup, results_dir=results_dir, resume=True) as run:
        return run_benchmark_setup(
            run=run, setup_name=setup, budget=budget,
            pop_size=ab.pop_size if pop_size is None else pop_size, evaluator=evaluator, space=space,
            max_workers=ab.max_workers if max_workers is None else max_workers,
            checkpoint_eval=int(budget * ab.share_checkpoint_fraction),
            checkpoint_interval=ab.checkpoint_interval if checkpoint_interval is None else checkpoint_interval,
            splits=splits)
