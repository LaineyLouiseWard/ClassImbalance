# Design notes

This note records the design decisions and negative results behind the experiments. The full
quantitative results are in the manuscript and `docs/NUMBERS.md`.

## Experimental design

The study is a 2×2 factorial over a fixed FT-UNetFormer backbone, crossing two data-curation
levers:

- **Cross-dataset transfer** — pre-train on a taxonomy-harmonised OpenEarthMap pool, then
  fine-tune on the Biodiversity training set (off / on).
- **Class-balanced sampling** — frequency-only class-balanced sampling (Kang et al., 2020) during
  training (off / on).

The four cells are baseline, transfer-only, sampler-only, and the full model (transfer plus
sampler), which is the deployed configuration. Each cell is trained over ten seeds, so effects are
reported with dispersion rather than from a single run.

## What the curation levers do

Both levers act on the data pipeline around a fixed backbone, so the 2×2 measures what curation alone
buys once the architecture is held constant. At ten seeds per cell, neither lever's main effect
separates from run-to-run variation: the design resolves roughly 3 percentage points of foreground
mean IoU across seeds, which is wider than the spread between the four cells. The honest reading is a
detection bound, not a null result — an effect smaller than that bound would not be visible with this
sample. The per-cell contrasts and their intervals are in the manuscript and `docs/NUMBERS.md`;
they are reported per seed, never from a pooled ensemble.

## Knowledge distillation (tested and dropped)

An earlier version of the pipeline added a distillation stage. It was tested against a step-matched
control that trained for the same number of additional steps without distillation. Distillation
underperformed that control, and no temperature or loss-weight setting recovered the difference. It
is therefore not part of the pipeline and is reported as a negative result. Self-distillation was
tested the same way, with the same outcome.

## Boundary-free tiles are excluded from every boundary analysis, not only from rho

**Decided 2026-07-28, recorded because it widens a pre-registered convention.**

The registration excludes tiles with no ground-truth boundary from rho, on the grounds that the
near-boundary set is empty by construction there and any error the tile holds can only depress the
ratio. `boundary_rate_ratio.py` has always done this. `boundary_trimap_iou.py` did not: a
single-class tile returns an all-infinite distance array, which `np.digitize` places in the deepest
interior bin, so the tile's entire foreground was counted as interior.

The exclusion is now applied at the top of that script's tile loop, which means it also removes
those tiles from the trimap IoU recovery curve and from Boundary-IoU — two analyses the registration
never named. That is deliberate: a tile with no boundary contributes to a boundary-tolerance curve
only by flattening it, and scoring the same ground under two different tile populations in one paper
is worse than widening one convention and saying so.

**What it changes.** Test A is unaffected (0 of 294 tiles). Test B loses 19 of 191 tiles, all
`ireland2`, holding 4,980,736 foreground pixels, or 16.17% of Test B foreground. Its interior (>8 m)
denominator falls from 23,968,766 to 18,988,030, a factor of 1.2623, so every Test B interior rate
this script reports rises by that factor. Train loses one tile (`biodiversity_0808`, 0.14%). These
counts match `artifacts/boundary_band_denominators.json` exactly.

**Also fixed at the same time:** the per-seed JSON key `rho` in that script held the 1.5 m
contact-zone ratio, not the 8 m registered statistic, and is renamed `contact_zone_ratio`; the
`d_m` field on the Boundary-IoU block converted a pixel radius at a hard-coded 0.5 m and is dropped,
since the uplands are 0.515 x 0.641 m and the band there is anisotropic.

## Conventions

- **Class order:** Background (0), Forest (1), Grassland (2), Cropland (3), Settlement (4),
  Seminatural (5), defined by `STUDENT_CLASSES` in `geoseg/taxonomy.py`.
- **Foreground mIoU:** the mean IoU over the five foreground classes (Background excluded), used
  for checkpoint selection and in all reported metrics.
- **Teacher model:** the OpenEarthMap teacher is built once and held fixed across the seed
  campaign; the OpenEarthMap→Biodiversity relabelling is grounded in the teacher's measured
  confusion.
