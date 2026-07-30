# Figures

The manuscript ships **five figures** plus a graphical abstract. Each is generated from a script in
`scripts/figures/` or `scripts/analysis/`. Script and output names are descriptive and stable; the
printed figure numbers are assigned by LaTeX, so the table below maps by content.

## Build

```bash
python scripts/figures/build_all_figures.py
```

This builds the five figures and copies them into the submission bundle (`manuscript_v2/Figures/`),
where `main.tex` reads them. TikZ figures are compiled with `pdflatex`; the matplotlib figures use
`text.usetex` (Latin Modern / Computer Modern), so a LaTeX toolchain is required — see the figure
prerequisites in [RUNBOOK.md](../RUNBOOK.md), section E.

Build a single figure by running its script directly, or skip figures with `--skip <name> ...`.

## Map

| Figure content | Source script | Output |
|----------------|---------------|--------|
| Study area and the spatially blocked split | `scripts/figures/study_area.py` | `study_area.pdf` |
| OpenEarthMap → Biodiversity taxonomy grounding | `scripts/figures/oem_mapping.tex` | `oem_mapping.pdf` |
| Foreground error by class pair, one factorial cell | `scripts/figures/pair_error_confusion.py` | `pair_error_confusion.pdf` |
| Whether the two grasslands meet: 8 m adjacency | `scripts/figures/class_seam.py` | `class_seam.pdf` |
| The two grasslands on the ground, four chips | `scripts/figures/two_grasslands_qualitative.py` | `two_grasslands_qualitative.pdf` |

The graphical abstract is built separately from `scripts/figures/graphical_abstract_panels.py` (three
raster panels) and `graphical_abstract_tikz.tex` (assembly).

## Exploratory figures

Several exploratory figures are not in the paper; their scripts remain under `scripts/figures/` and
`scripts/analysis/` and can be run by hand, but they are out of `build_all_figures.py`. They include
a two-axis mitigation schematic, class distributions, a frequency-versus-difficulty panel, an
ablation-qualitative comparison, boundary-limited error curves, and residual-uncertainty figures.
Several read inputs that were archived when the withdrawn 2026-07-25 campaign was cleared away.

Per-figure inputs (data paths, checkpoints, evaluation outputs) are documented stage-by-stage in
[RUNBOOK.md](../RUNBOOK.md). Figures that draw on the proprietary Biodiversity imagery cannot be
regenerated without licensed access to that dataset — see the data-availability note in the
[README](../README.md).
