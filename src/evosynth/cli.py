"""
`evosynth` command-line interface (also: `python -m evosynth`).

    evosynth run        [--full | --preset NAME] [--dataset covertype] [--budget N ...] [--no-refit]
    evosynth scale      --budget 300 [--pop-size N] [--checkpoint-interval N]
    evosynth ablation   [--setup CVGS-500] [--pop-size N] [--max-workers N]
    evosynth baselines  [--test all|extratrees|random|optuna|flaml] [--trials N] [--time-budget S]
    evosynth nas
    evosynth serve      [--host H] [--port P]
    evosynth settings   [show [SECTION] | check] [--preset NAME]
    evosynth datasets   list

Every command also accepts:
    --set SECTION.KEY=VALUE   override any setting for this run (repeatable), e.g.
                              --set bandit.c=0.2 --set optimizers.pso.w=0.7
    --settings PATH           use another settings folder (or a single .toml file)

Commands that run experiments also accept:
    --seed N                  search seed for this run (sets run.seed)
    --seeds 1,2,3             repeat the run once per seed, one seed_<k> folder each
    --variant NAME            name of the run folder level (default: preset / budget_<N> / setup)
    --results-dir PATH        write under PATH instead of results/

Each run writes to results/<experiment>/<dataset>/<variant>/seed_<run.seed>/
(manifest.json, result.json, run.log, events.jsonl, ...).

Settings are resolved as: settings/*.toml -> preset -> command-line flags -> --set.
"""
import argparse
import json
import sys

from evosynth.config import ConfigError, load_config, set_settings
from evosynth.telemetry.run_logger import make_safe


# ---------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------
def _common(parser):
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="override a setting for this run (repeatable)")
    parser.add_argument("--settings", default=None, metavar="PATH",
                        help="settings folder or .toml file (default: settings/)")


def _activate(args, preset=None, extra=None):
    """Load settings (+ preset, flag-derived overrides, --set) and make them active."""
    overrides = list(extra or []) + list(args.overrides)
    cfg = load_config(path=args.settings, preset=preset, overrides=overrides)
    set_settings(cfg)
    return cfg


def _seed_list(args):
    """[None] (use run.seed from settings), [N] for --seed N, or the --seeds list."""
    if getattr(args, "seeds", None):
        try:
            return [int(x) for x in args.seeds.split(",") if x.strip()]
        except ValueError:
            raise ConfigError(f"--seeds must be a comma-separated list of integers (got {args.seeds!r})")
    return [getattr(args, "seed", None)]


def _per_seed(args, body, preset=None, extra=None):
    """Activate the settings and call body() once per requested seed (sequentially)."""
    seeds = _seed_list(args)
    for i, seed in enumerate(seeds):
        seed_override = [] if seed is None else [f"run.seed={seed}"]
        cfg = _activate(args, preset=preset, extra=list(extra or []) + seed_override)
        if len(seeds) > 1:
            print(f"\n##### seed {cfg.run.seed} ({i + 1}/{len(seeds)}) #####", flush=True)
        body()


def _flag_overrides(args, mapping):
    """['engine.budget=120', ...] for every flag in mapping that was given."""
    out = []
    for attr, key in mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            out.append(f"{key}={value}")
    return out


# ---------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------
def cmd_run(args):
    preset = "search_full" if args.full else args.preset
    extra = _flag_overrides(args, {"budget": "engine.budget", "pop_size": "engine.pop_size",
                                   "t1_iters": "engine.t1_iters", "k_iters": "engine.k_iters"})
    from evosynth.experiments.search import run_search
    label = "Full Benchmark" if preset == "search_full" else ("Fast Demo" if preset == "search_demo" else preset)
    _per_seed(args, lambda: run_search(dataset=args.dataset, mode_label=label, subsample=args.subsample,
                                       refit=not args.no_refit, variant=args.variant or preset,
                                       results_dir=args.results_dir),
              preset=preset, extra=extra)


def cmd_scale(args):
    from evosynth.experiments.scaling import run_benchmark
    _per_seed(args, lambda: run_benchmark(budget=args.budget, pop_size=args.pop_size,
                                          checkpoint_interval=args.checkpoint_interval, variant=args.variant,
                                          results_dir=args.results_dir, dataset=args.dataset))


def cmd_ablation(args):
    from evosynth.experiments.ablation import run_ablation
    _per_seed(args, lambda: run_ablation(setup=args.setup, pop_size=args.pop_size, max_workers=args.max_workers,
                                         checkpoint_interval=args.checkpoint_interval,
                                         results_dir=args.results_dir, dataset=args.dataset))


def cmd_baselines(args):
    from evosynth.experiments.baselines import run_baselines
    _per_seed(args, lambda: run_baselines(test=args.test, trials=args.trials, time_budget=args.time_budget,
                                          results_dir=args.results_dir))


def cmd_nas(args):
    from evosynth.experiments.nas import run_nas
    _per_seed(args, lambda: run_nas(variant=args.variant or "default", results_dir=args.results_dir))


def cmd_serve(args):
    _activate(args)
    from evosynth.telemetry.server import serve
    serve(host=args.host, port=args.port)


def cmd_settings(args):
    cfg = _activate(args, preset=args.preset)
    if args.action == "check":
        files = "settings/" if args.settings is None else args.settings
        print(f"settings OK ({files}): {len(cfg)} sections - " + ", ".join(sorted(cfg)))
        return
    data = cfg.to_dict()
    if args.section:
        for key in args.section.split("."):
            if not isinstance(data, dict) or key not in data:
                raise ConfigError(f"no setting/section '{args.section}'")
            data = data[key]
    print(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_datasets(args):
    _activate(args)
    from evosynth import datasets
    from evosynth.config import get_settings
    for name in datasets.available():
        spec = get_settings().datasets[name]
        print(f"{name:<12} {spec.get('display_name', '')}  [source: {spec.source}, metric: {spec.metric}]")


# ---------------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(prog="evosynth", description="EvoSynth dynamic meta-optimizer",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="one search + refit (presets: search_demo, search_full)")
    r.add_argument("--preset", default="search_demo", help="[experiments.<preset>] to apply (default: search_demo)")
    r.add_argument("--full", action="store_true", help="shortcut for --preset search_full")
    r.add_argument("--dataset", default="covertype")
    r.add_argument("--budget", type=int)
    r.add_argument("--pop-size", "--pop_size", dest="pop_size", type=int)
    r.add_argument("--t1-iters", "--t1_iters", dest="t1_iters", type=int)
    r.add_argument("--k-iters", "--k_iters", dest="k_iters", type=int)
    r.add_argument("--subsample", type=int, help="search rows (default: datasets.<name>.search_subsample)")
    r.add_argument("--no-refit", action="store_true", help="skip the post-search refit")
    r.set_defaults(func=cmd_run)

    sc = sub.add_parser("scale", help="budget-scaling run with checkpointing + refit")
    sc.add_argument("--budget", type=int, help="one of experiments.scaling.budgets (default: default_budget)")
    sc.add_argument("--pop-size", "--pop_size", dest="pop_size", type=int)
    sc.add_argument("--checkpoint-interval", "--checkpoint_interval", dest="checkpoint_interval", type=int)
    sc.add_argument("--dataset", default="covertype")
    sc.set_defaults(func=cmd_scale)

    a = sub.add_parser("ablation", help="isolated optimizers vs EvoSynth")
    a.add_argument("--setup", help="key of experiments.ablation.setups (default: first)")
    a.add_argument("--pop-size", "--pop_size", dest="pop_size", type=int)
    a.add_argument("--max-workers", "--max_workers", dest="max_workers", type=int)
    a.add_argument("--checkpoint-interval", "--checkpoint_interval", dest="checkpoint_interval", type=int)
    a.add_argument("--dataset", default="covertype")
    a.set_defaults(func=cmd_ablation)

    b = sub.add_parser("baselines", help="default ExtraTrees, Random Search, Optuna TPE, FLAML")
    b.add_argument("--test", default="all", choices=["all", "extratrees", "random", "optuna", "flaml"])
    b.add_argument("--trials", type=int, help="Random Search / Optuna trials")
    b.add_argument("--time-budget", "--time_budget", dest="time_budget", type=int, help="FLAML seconds")
    b.set_defaults(func=cmd_baselines)

    n = sub.add_parser("nas", help="CNN architecture search demo")
    n.set_defaults(func=cmd_nas)

    sv = sub.add_parser("serve", help="start the telemetry relay for the live dashboard")
    sv.add_argument("--host")
    sv.add_argument("--port", type=int)
    sv.set_defaults(func=cmd_serve)

    st = sub.add_parser("settings", help="show or check the resolved settings")
    st.add_argument("action", nargs="?", default="show", choices=["show", "check"])
    st.add_argument("section", nargs="?", help="e.g. engine or optimizers.pso")
    st.add_argument("--preset", default=None)
    st.set_defaults(func=cmd_settings)

    d = sub.add_parser("datasets", help="list registered datasets")
    d.add_argument("action", nargs="?", default="list", choices=["list"])
    d.set_defaults(func=cmd_datasets)

    for sp in (r, sc, a, b, n, sv, st, d):
        _common(sp)
    for sp in (r, sc, n):
        sp.add_argument("--variant", help="run folder name (default: preset / budget_<N> / default)")
    for sp in (r, sc, a, b, n):
        seeds = sp.add_mutually_exclusive_group()
        seeds.add_argument("--seed", type=int, help="search seed (sets run.seed)")
        seeds.add_argument("--seeds", metavar="LIST", help="comma-separated seeds; one run (and folder) per seed")
        sp.add_argument("--results-dir", "--results_dir", dest="results_dir", metavar="PATH",
                        help="write under PATH instead of results/")
    return p


def main(argv=None):
    make_safe(sys.stdout)
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except ConfigError as exc:
        print(f"evosynth: settings error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
