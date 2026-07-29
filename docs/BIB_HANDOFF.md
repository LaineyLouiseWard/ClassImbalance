# Bibliography handoff — 2026-07-29

Everything below is verified. Anything that failed verification has been removed rather than hedged.

## Already done — no action needed

`manuscript_v2/Bibliography.bib` is **59 entries, 25 cited, renders 0 warnings**.

- 25 bibliography fixes applied (details: `docs/BIB_AUDIT_2026-07-29.md`).
- Roscher cut — it was uncited.
- `Bibliography_additions.bib` deleted; Krawczyk and Maxwell were already merged.
- Every cited source now has a conversion in `references_md/`, indexed in `docs/SOURCE_CONVERSIONS.md`.

## Action 1 — the correlogram sentence (Methods)

**Do not cite Legendre & Legendre, Legendre & Gallagher, Mantel 1967, or Bjørnstad.** None is held in
any library on this machine — checked across both Zotero libraries (9,376 items), 1,339 stored PDFs
and all three conversion directories. Unread, therefore barred.

**Do not lean on Kattenborn for the measurement either.** He is a precedent for the *approach* only.
His own scripts run `correlog(..., increment=1, resamp=10)` — ten permutations, no usable significance
test, no range estimated, and his blocks were whole sites chosen a priori. He did not measure a range
and size blocks from it; we did.

**Cite Roberts**, who states the rule directly and is converted and read. Proposed sentence:

> Block and buffer widths follow Roberts et al., who set them from the measured range of dependence
> in the data rather than by eye \cite{robertsCrossValidationStrategies2017}; that range was estimated
> per site with a multivariate Mantel correlogram over 100 m distance classes, computed on per-tile
> class composition and repeated on per-tile band statistics as a check, which puts the inland range
> at 750 m at a peak Mantel r of 0.044 — a separation the 1,664 m train–test gap clears and the 256 m
> train–validation gap does not.

**Do not write that the analysis "mirrors" Kattenborn.** His response descriptor is close to ours
(per-tile class proportions), but his predictor descriptor is 200 VAE latents chosen specifically
because the reduction must "respect the spatial structure of data", whereas our per-band means and
percentiles are permutation-invariant within a tile. The analogy does not hold in the manuscript.

## Action 2 — no AdamW citation

Describe the optimiser in Methods. Note `loshchilov-2017-sgdr-warm-restarts` in the library is the
**scheduler** paper, already cited — not AdamW.

## Action 3 — citation count

25 cited is lean. The best additions are already in the `.bib` as dead entries and already converted:

| Add | Why |
|---|---|
| `bressanSemanticSegmentationLabeling2022` | Segmentation + labelling uncertainty + class imbalance + vegetation. Closest prior work in the file, and "what we cannot settle" currently cites nothing |
| `waldnerNeedleHaystackMapping2019` | Rare class, satellite imagery, data balancing — the missing precedent for the oversampling factor |
| `chenClassImbalanceAutomatic2025` **or** `sharmaAddressingClassImbalance2025` | Remote-sensing-specific imbalance review |
| `gevaertAuditingGeospatialDatasets2024` | Auditing a dataset rather than changing the model — this paper's practitioner move |

**Do not add** `oksuz` (declares its own scope as object detection; citing it while two RS-specific
imbalance reviews sit uncited invites a reviewer question), or `haixiang` / `park` (redundant against
Krawczyk and Lin). Avoid the uncertainty-quantification and distillation blocks entirely — leftovers
from dropped directions.

## Action 4 — re-check every Cheng quotation

The Cheng conversion in use until today was the **arXiv version wearing a CVPR label**
(`venue: "CVPR (converted from arXiv 2103.16562)"`). `docs/DO_NOT_ADD.md` requires the proceedings.
The genuine proceedings conversion is now `references_md/cheng-2021-boundary-iou-cvpr-proceedings.md`.
Any Cheng quotation predating 2026-07-29 rests on the wrong file.

## Still outstanding — the eight blockers

`docs/CLAIM_SUPPORT_AUDIT_2026-07-29.md`, top section. The two worst are not citation problems:

1. **Results §3.3 contradicts itself** — reports semi-natural wrong 27.0% beyond 32 m, then says
   "Both grassland classes concentrate error near boundaries; only forest and grassland clear it deep
   inside."
2. **Contributions** asserts "Because the error is not concentrated at boundaries" without the
   qualifier; §3.3 gives 3.7× within one metre and says only the grassland pair is the exception.

## One code finding, not blocking

`scripts/analysis/spatial_correlogram.py`'s progressive Holm **never sorts the p-values**. On a
non-monotone sequence it returns 1.0 where `vegan` returns 0.002. It was mutation-tested: **every
stored range reproduces exactly, so the published numbers stand.** Worth fixing before reuse. The
script also uses a two-tailed p where vegan uses one-tailed.
