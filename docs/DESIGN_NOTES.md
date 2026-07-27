> **⚠ STALE BY DECISION — 2026-07-27.** Every accuracy, contrast and figure below comes from a
> campaign **withdrawn on 2026-07-25** for train/test leakage: tiles are chipped on a 50% stride and
> the split was random by tile, so ~93% of each held-out tile's ground was also in training. The
> split described here (1,706/219/218) no longer exists. Treat every number as ABSENT, not
> provisional. The current design is in `docs/METHODOLOGICAL_CHOICES.md`; this file is rewritten
> after the rebuilt campaign runs.

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

## Conventions

- **Class order:** Background (0), Forest (1), Grassland (2), Cropland (3), Settlement (4),
  Seminatural (5), defined in `geoseg/datasets/biodiversity_dataset.py`.
- **Foreground mIoU:** the mean IoU over the five foreground classes (Background excluded), used
  for checkpoint selection and in all reported metrics.
- **Teacher model:** the OpenEarthMap teacher is built once and held fixed across the seed
  campaign; the OpenEarthMap→Biodiversity relabelling is grounded in the teacher's measured
  confusion.
