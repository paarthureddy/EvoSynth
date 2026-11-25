# Paper and presentation (Group G002)

| File | What it is |
|---|---|
| `paper.tex` | Conference paper (IEEEtran). Formerly `submission/G002_Paper.tex`. |
| `slides.tex` | Panel Review slides (Beamer). Formerly `submission/maint.tex`. |
| `figures/` | Images used by both: `arch.png` (architecture), `logo.png`. Same images as `docs/public/`. |
| `links.md` | Dataset DOI, repository and documentation site. |
| `panel_review.pdf` | Slides as presented at the Panel Review (local only, git-ignored). |

Build from this folder, e.g. `pdflatex paper.tex` and `pdflatex slides.tex`.

**Missing figures:** `slides.tex` uses `approval.png` and `arch_updated.png`, which are not in the
repository. Put them into `figures/` before building the slides.

`paper.tex` still describes the v1 (panel-review) code: its Covertype numbers match the budget-120
run in `results/archive/2026-11-19_covertype_v1/evosynth/run_120.txt`, and it still contains the
pathological-landscape experiment and the digits NAS demo. All of this is rewritten after the
re-runs (Phase 6).
