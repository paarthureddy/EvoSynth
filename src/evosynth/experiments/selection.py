"""
Confirm before select (Phase 3.7).

A search's best score is the maximum over many noisy cv_folds-fold estimates, so the
"winner" is partly the configuration that got the luckiest folds. Before the refit, the top_k
distinct configurations (by search score) are re-scored with confirm_folds-fold CV on the same
search rows, and the one with the best confirmed score is selected.

Settings: [selection] enabled, top_k, confirm_folds (settings/evaluation.toml).
"""
import json

from evosynth.config import get_settings
from evosynth.core.decoding import decode_vector_to_dict


def _key(cfg):
    return json.dumps(cfg, sort_keys=True, default=str)


def top_distinct(individuals, space, k):
    """The k best individuals whose decoded configurations differ, best first, as (ind, cfg)."""
    out, seen = [], set()
    for ind in sorted(individuals, key=lambda i: i.fitness, reverse=True):
        cfg = decode_vector_to_dict(ind.vector, space)
        key = _key(cfg)
        if key not in seen:
            seen.add(key)
            out.append((ind, cfg))
            if len(out) == k:
                break
    return out


def confirm_best(evaluator, individuals, space, settings=None):
    """
    Returns (selected Individual, selected config, report). report is None when [selection] is
    disabled; otherwise a dict for result.json with every candidate's search and confirmed score.
    """
    s = (settings or get_settings()).selection
    best = max(individuals, key=lambda i: i.fitness)
    if not s.enabled:
        return best, decode_vector_to_dict(best.vector, space), None

    candidates = top_distinct(individuals, space, s.top_k)
    print(f"\n  [CONFIRM] Re-scoring the top {len(candidates)} distinct configurations with "
          f"{s.confirm_folds}-fold CV (not counted in the budget)...")
    rows = []
    for rank, (ind, cfg) in enumerate(candidates, 1):
        confirmed = evaluator.evaluate_config(cfg, s.confirm_folds)
        rows.append({"search_rank": rank, "search_score": float(ind.fitness),
                     "confirmed_score": float(confirmed), "source_algorithm": ind.source_algorithm,
                     "config": cfg})
        print(f"    #{rank}  search {ind.fitness:6.2f}%  ->  confirmed {confirmed:6.2f}%  "
              f"({cfg.get('model_type')}, fe={cfg.get('fe_strategy')}, by {ind.source_algorithm})")
    pick = max(range(len(rows)), key=lambda i: rows[i]["confirmed_score"])
    print(f"  [CONFIRM] Selected search rank #{rows[pick]['search_rank']} "
          f"(confirmed {rows[pick]['confirmed_score']:.2f}%).")
    report = {"top_k": s.top_k, "confirm_folds": s.confirm_folds, "selected_search_rank": pick + 1,
              "candidates": rows}
    return candidates[pick][0], candidates[pick][1], report
