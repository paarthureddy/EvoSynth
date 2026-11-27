# Paper, report and presentation (Group G002)

| File | What it is |
|---|---|
| `paper.tex` | Conference paper (IEEEtran): method, and the Phase 5 results on Covertype and Adult (main comparison, ablation, speed-up, inference cost, Fashion-MNIST NAS). |
| `report.tex` | Project Phase II (External Review) report in the Amrita format, including the individual certifications. |
| `slides.tex` | Review slides (Beamer), on the department's review template. |
| `figures/` | Images: `arch.png` (current architecture; source `arch_source.html`, rendered with headless Chrome), `arch_legacy.png` (the panel-review architecture), `logo.png`. Same images as `docs/public/`. |
| `links.md` | Dataset DOI, repository and documentation site. |
| `panel_review.pdf` | Slides as presented at the Panel Review (local only, git-ignored). |

Build from this folder with any LaTeX distribution (MiKTeX, TeX Live, Tectonic) or Overleaf:
`latexmk -pdf paper.tex`, `latexmk -pdf report.tex`, `latexmk -pdf slides.tex`.

**Missing figure:** `slides.tex` uses `approval.png`, which changes for every review and is not
kept in the repository. Put it into `figures/` (or next to `slides.tex`) before building the slides.

All numbers come from `results/summary/` (seed 42); see [`handbook/08-results.md`](../handbook/08-results.md)
and [`handbook/10-paper.md`](../handbook/10-paper.md). The panel-review (v1) version of the paper is
in git history (tag `v1-panel-review`, local).
