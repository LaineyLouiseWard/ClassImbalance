# Figures

Seven figures are currently built; the rebuilt manuscript ships **three or four** (see
`notes/IMPLEMENTATION_PLAN.md`). The "thirteen figures" this line used to claim belonged to the
withdrawn campaign. Each is generated from a script in `scripts/figures/`
or `scripts/analysis/`. Script and output names are descriptive and stable; the printed figure
numbers are assigned by LaTeX, so the table maps by content.

## Build

```bash
python scripts/figures/build_all_figures.py
```

This builds all seven figures and copies them into the submission bundle
(`manuscript/Figures/`), which is where `main.tex` reads them. TikZ figures are compiled with
`pdflatex`; the matplotlib figures use `text.usetex` (Latin Modern / Computer Modern), so a LaTeX
toolchain is required — see the figure prerequisites in [RUNBOOK.md](../RUNBOOK.md), section E. `boundary_limited_error.py`
lives in `scripts/figures/` and depends on saved per-tile evaluation outputs. (This paragraph named
three figures under `scripts/analysis/` until 2026-07-29; the uncertainty and cross-check figures are
both on the cut list below, and `run_analysis()` in `build_all_figures.py` is now dead code.)

Build a single figure by running its script directly, or skip figures with
`--skip <name> ...` (for example `--skip ablation_qualitative uncertainty_overlay`).

## Map

**Six figures** as of 2026-07-29. See the cut list below and the 2026-07-29 note.

| Figure content | Source script | Output |
|----------------|---------------|--------|
| Study area | `scripts/figures/study_area.py` | `study_area.pdf` |
| Ablation qualitative comparison | `scripts/figures/ablation_qualitative.py` | `ablation_qualitative.pdf` |
| Foreground error by class pair, one factorial cell | `scripts/figures/pair_error_confusion.py` | `pair_error_confusion.pdf` |
| The two grasslands on the ground, four chips | `scripts/figures/two_grasslands_qualitative.py` | `two_grasslands_qualitative.pdf` |
| Whether the two classes meet: 8 m adjacency | `scripts/figures/class_seam.py` | `class_seam.pdf` |
| Boundary-limited error | `scripts/figures/boundary_limited_error.py` | `boundary_limited_error.pdf` |

### Changed 2026-07-29

`confusion_matrices.py` was renamed `pair_error_confusion.py` and cut from three panels
(baseline / full / delta) to one factorial cell — the factorial is a bound on what the design can
resolve, not the finding, and several near-identical matrices invited the reader to hunt for
differences the design cannot separate. `confusion_matrices.pdf` no longer exists.
`two_grasslands_qualitative` and `class_seam` are new. `class_distributions`, `workflow_pipeline`
and `oem_mapping` left the build.

**All three new figures are built 7.28 in wide** for `\begin{adjustwidth}{-\extralength}{0cm}`
(`\textwidth` 13.90 cm + `\extralength` 4.61 cm), not for plain `\textwidth`. Placing one in a
`\textwidth` figure shrinks every label by a quarter. Handover notes for the manuscript chat, with
the selection rules each caption has to state, are in `notes/FIGURES_FOR_MAIN_CHAT.md`.

## Cut on 2026-07-28, and why

Six figures were removed from the build. Their scripts remain and can still be run by hand; they are
out of `build_all_figures.py` because four of them read inputs that moved to
`../_archive-lqc/withdrawn_campaign_2026-07-28/`, so the build raised `SystemExit` and refused to sync even
the figures that had succeeded.

| figure | why |
|---|---|
| Two-axes mitigation schematic | draws a data-versus-model split the fixed architecture cannot test |
| Frequency vs difficulty | five classes cannot support the claim in either direction |
| Reliability / ECE | residual-uncertainty section, cut |
| Uncertainty quality | same |
| Uncertainty overlay | same, and its default tiles are now in train and test |
| Confident-learning cross-check (appendix) | headline was a retracted statistic, and its noise assumption is violated by the spatial structure the paper claims |

The graphical abstract is built separately from `scripts/figures/graphical_abstract_panels.py`
(three raster panels) and `graphical_abstract_tikz.tex` (assembly); the final image ships with the
submission bundle.

Per-figure inputs (data paths, checkpoints, evaluation outputs) are documented stage-by-stage in
[RUNBOOK.md](../RUNBOOK.md). Figures that draw on the proprietary Biodiversity imagery cannot be
regenerated without licensed access to that dataset — see the data-availability note in the
[README](../README.md).
