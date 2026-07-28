> **⚠ STALE BY DECISION — 2026-07-27.** Every accuracy, contrast and figure below comes from a
> campaign **withdrawn on 2026-07-25** for train/test leakage: tiles are chipped on a 50% stride and
> the split was random by tile, so ~93% of each held-out tile's ground was also in training. The
> split described here (1,706/219/218) no longer exists. Treat every number as ABSENT, not
> provisional. The current design and results are in `docs/README.md`, which says what to read and in
> what order; this file is rewritten after the manuscript is finished.
>
> **Scope of this banner:** it covers "Experimental design", "What the curation levers do" and
> "Knowledge distillation" below. It does NOT cover "Boundary-free tiles are excluded from every
> boundary analysis", which is dated 2026-07-28 and current.

# Design notes

This note records the main design decisions behind the experiments and the results that did not
support a mechanism. It complements the manuscript, which carries the full quantitative results.

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

On a strong pre-trained backbone the rare classes are already recovered to a large degree, so the
curation levers move the result only modestly. Cross-dataset transfer gives a small, consistent
gain; class-balanced sampling adds little once transfer is in place. This is the finding of the
study, not a shortcoming: the remaining error is concentrated at class boundaries and reflects the
quality of the labels rather than model capacity or the sampling scheme.

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
  Seminatural (5), defined in `geoseg/datasets/biodiversity_dataset.py`.
- **Foreground mIoU:** the mean IoU over the five foreground classes (Background excluded), used
  for checkpoint selection and in all reported metrics.
- **Teacher model:** the OpenEarthMap teacher is built once and held fixed across the seed
  campaign; the OpenEarthMap→Biodiversity relabelling is grounded in the teacher's measured
  confusion.
