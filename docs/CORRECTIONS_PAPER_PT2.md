# Corrections, part 2 — overlapping tiles in the scored splits

Found 2026-07-27, after the split rebuild. Separate from `CORRECTIONS.md`, which covers sentences
already known to be wrong. Everything here is new and none of it is in the manuscript yet.

---

## In plain language, before the detail

Two problems were found on 2026-07-27. Both are fixed. Neither is in the manuscript yet.

**Problem 1 — the brightness ruler was measured per tile.** Before an image goes into the model,
its brightness is rescaled onto a 0-255 range. That rescaling was worked out separately for every
tile, using that tile's own darkest and brightest pixels. Tiles overlap by half, so the same field
sits in up to four tiles and got four different rescalings. The model then saw the same ground four
different ways and sometimes gave four different answers.

That matters more than it sounds, because of *where* the disagreements land. In the middle of a
field the model is 95% sure, so nudging the brightness changes nothing. At the edge between two
land-cover types it is nearly 50/50, so the same nudge tips it over. Measured: predictions flipped
on 15.2% of pixels within half a metre of a label boundary, against 0.52% in field interiors — 29x.
That is the same pattern the paper attributes to ambiguous labels, produced by an arbitrary
preprocessing choice instead.

**Fixed** by measuring the ruler once per site rather than once per tile. Proof: overlapping tiles
now produce byte-identical pixel values on the ground they share (0.00% differ; it was 91-99.8%).
The paper needs one sentence in §2; nothing else changes, and the boundary analysis is untouched.

**Problem 2 — scoring counted some ground several times.** The scorer added up one confusion matrix
per tile. Because tiles overlap, a field in the middle of the test strip entered the total about
four times and one at the edge once. Worse, the over-counting differed between Test A (2.85x) and
the two upland sites (3.05x, 3.26x) — and the Test A vs Test B comparison is the paper's headline.

**Fixed** by scoring only tiles that do not overlap each other: 90 of 294 on Test A, 51 of 191 on
Test B. No averaging and no combining — the duplicate photographs of already-scored ground are
simply not used. It keeps 90% and 93% of the labelled ground, with every class retained evenly.

Two alternatives were tried and rejected, both because they changed more than the double counting:
averaging the overlapping predictions is a hidden ensemble (it makes the model look better without
being better), and taking each pixel from the tile it sits most centrally in changes which part of
each tile gets scored, by different amounts per site.

**Worth knowing, not alarming.** The subset leaves Test A's Cropland thinnest at 0.073 km²
(~293,000 labelled pixels), so its per-class IoU will be the noisiest — say so in the results rather
than be asked. And tile counts flatter the test sets generally: report ground area instead (§6).

---

## 1. The problem

Tiles are 256 m footprints on a 128 m grid, so every tile overlaps its neighbours by half. The
rebuilt split fixed **cross-split** overlap (verified zero, recomputed from GeoTIFF bounds). It did
not remove overlap **within** each scored split. `evaluation/compute_metrics.py:218-221` accumulates
`cm += tile_cm` per tile, so a ground pixel in the middle of the test strip enters the confusion
matrix about four times and a pixel at the strip edge once. The metric is over tiles, not over ground.

Verified geometry: Test A is 294 tiles, all 512×512 at exactly 0.5 m, on a clean lattice — 7 unique
x positions, 52 unique y, every step exactly 128 m, one projected CRS. The two upland sites of Test B
are stored in a geographic CRS (bounds in degrees; this is the source of the 0.515 × 0.641 m
anisotropy).

## 2. What the literature does

Two accepted treatments, and this pipeline does neither:

- **Non-overlapping scored tiles.** Cira et al. 2024, *Remote Sensing* 16(16) 2954 — the target
  journal — overlap train and validation (12.5%) and cut the test area *"with no overlap… and compute
  the models' performance metrics"*. Every test row in their Table 1 is labelled "no overlap".
- **Merge overlapping predictions, then score once.** Reina et al. 2020 (already in
  `references_md/`): *"Inference was performed using tiles… with a 50% overlap… The overlapping tiles
  were averaged to provide the whole image prediction."*

GeoSeg (the FT-UNetFormer authors' repo, which this code derives from) uses the same 50% stride for
train, val and test and scores tile by tile. That is a leaderboard convention — everyone shares the
protocol so the ranking stays consistent — and it is not a justification for a paper making claims
about where on the ground error sits.

## 3. Decision — score a non-overlapping subset (2026-07-27, SETTLED)

**Keep only tiles that do not overlap each other, and score those.** Greedy packing on the real
lattice: Test A 294 -> 90 tiles, ireland1 64 -> 18, ireland2 127 -> 33. No new estimator, no
averaging, no weighting — a list of filenames. Precedent: Cira et al. 2024 in the target journal cut
their test area with no overlap.

Measured cost, deduplicated labelled ground: **Test A keeps 5.20 of 5.76 km² (90%), Test B keeps
2.44 of 2.62 km² (93%)**. Per-class retention is uniform — 28-31% of each class on Test A, 25-29% on
ireland1, 23-24% on ireland2 — so it is an unbiased spatial thinning, not a selective one. No class
is starved. (ireland2 shows 0% Cropland because it contains none at all.)

**Rejected, with the measurement that killed each:**

- *Average the overlapping softmax.* It is a 3-4 member test-time ensemble, not de-duplication:
  the per-tile stretch made the overlapping predictions genuinely different predictors. On a
  synthetic case with a known 25% error rate, `nearest` scored 0.7498 pixel accuracy (tracking the
  true rate) while `mean` scored 0.8922 — a +0.142 uplift that is not a correction. *Note the
  normalisation fix weakens but does not remove this: identical inputs still enter the network at
  different positions within their crop, and models are not translation-invariant (Reina et al.
  2020 measure exactly this).*
- *Nearest tile centre / centre cropping.* One prediction per pixel, so no ensemble — but it changes
  the share of scored ground taken from tile margins from 75.4% to 29.9% (Test A) and 19.9-25.0%
  (uplands), **by different amounts per site**. Margin predictions are worse, so it biases upward
  and it distorts the Test A vs Test B contrast, which is the paper's headline.

Only the subset changes the double-counting and nothing else.

## 3b. Superseded — the stitching approach, kept for the record

Every number reported in the paper comes from a map on which each piece of ground is counted exactly
once. Implemented in `scripts/analysis/stitch_predictions.py`. Reina et al. 2020 is the precedent.

**Reported aggregation is `nearest`: each ground pixel takes the prediction of the one tile whose
centre is nearest.** Not the mean of the overlapping softmax — and this distinction is load-bearing.
`_normalize_percentile` (`geoseg/datasets/biodiversity_dataset.py:53-63`) computes 2-98 percentiles
**per tile**, so the same ground pixel reaches the network under up to four different contrast
stretches. Averaging them is a genuine 3-4 member test-time ensemble stacked on top of the
de-duplication, and on top of the TTA already applied at stage C2. It would raise the reported score
for a reason unrelated to the defect being fixed.

Measured on a synthetic case with a known 25% per-tile error rate: `nearest` scores 0.7498 pixel
accuracy, tracking the true rate; `mean` scores 0.8922. **That +0.142 is ensemble uplift, not a
correction.** `mean` and `vote` are retained behind `--cross-check` as diagnostics only.

Applies to Test A, Test B, and any validation number that reaches the paper. Training is untouched —
overlapping training chips are correct and deliberate. Checkpoint selection on validation is
untouched: overlap weights ground unevenly but identically across all four cells and ten seeds, so it
cannot bias which epoch wins.

Assembly differs per split, scoring does not. Test A and val are `EPSG:32629`, 0.5 m square pixels —
exact pixel placement, no resampling. Test B is `EPSG:4326`, 9.26e-6° x 4.63e-6° (0.515 x 0.641 m on
the ground) — assembled on its own lat/lon canvas. Both maps are then scored by the same single pass.
Test B's pixels are not equal-area, but that is already true of how it is scored today; stitching does
not make it worse.

**Rejected alternative:** score a non-overlapping subset of tiles (greedy packing keeps 90 of 294,
covering 5.90 of 6.77 km²). Zero new code, but it discards 13% of the ground and does not fix the
rho bias in §4. Kept on record because it is the fallback if the assembly code cannot be trusted in
time.

Runs locally after the campaign, from the stage C5 softmax dumps. Nothing here blocks the launch.
`RUNBOOK.sh` C5 now also dumps validation softmax, as insurance — without it, wanting any validation
number later means re-running inference on all forty checkpoints.

## 4. New limitation the paper must state

Tiled inference degrades boundaries independently of label quality. Reina et al. 2020 show tiled
predictions are *"blobbier or more amorphous"* and *"fail to capture fine-grained boundaries"*, with
whole-image inference scoring 0.917 Dice against 0.791 for 128 px tiles on satellite data; the cure
is larger tiles, not averaging.

This is a **third** rival explanation for boundary-concentrated error, alongside the two §4.5 already
names (label ambiguity, encoder–decoder edge blur). Add it. It is constant across all four cells, so
it touches no contrast — only the absolute claim.

Second, related: `boundary_distance(mask, tile_id)` is computed per tile, so a class boundary just
outside a tile is invisible and pixels near the tile edge are assigned to the **interior** stratum
when they are genuinely near a boundary. Those pixels carry high error, which inflates the interior
rate — rho's denominator. **rho is therefore biased downward.** State it; a conservative bias is
defensible, an unstated one is not.

## 5. Two descriptive statistics affected

Both are inflated by overlap, because one patch of a rare class appears in up to four tiles.

- `main.tex:152` — semi-natural "spread over 614 tiles".
- Figure `class_distributions`, panels (a,c) — "proportion of tiles containing each class".

Neither is load-bearing. Recompute on ground, or reword away from tile counts.

## 6. Dataset facts found 2026-07-27 that the paper currently misstates or omits

All measured here, deduplicated for tile overlap, using per-site GSD from `geoseg.geo.GSD_BY_SITE`.

| split | tiles | ground covered | **labelled** | labelled % |
|---|---|---|---|---|
| train | 1072 | 23.28 km² | 22.58 km² | 97.0% |
| val | 173 | 4.26 km² | 4.17 km² | 97.9% |
| Test A | 294 | 6.77 km² | 5.76 km² | 85.2% |
| Test B | 191 | 5.16 km² | **2.62 km²** | **50.7%** |

- [ ] **Report ground area, not tile counts.** Tile counts overstate: 22 of the 191 upland tiles are
  >90% void and 55 are >50% void, because they straddle the edge of a surveyed farm. "2.62 km² of
  labelled ground" is the honest figure and is still larger than the whole test set of either
  benchmark this literature is validated on (ISPRS Potsdam ≈1.26 km², Vaihingen ≈0.71 km²).
- [ ] **State that Test B is only half labelled.** Its extent covers 5.16 km² but only 2.62 km² sits
  inside surveyed farms. One clause in §2.1.2.
- [x] **`ireland2` is stored on a different scale — RESOLVED 2026-07-27, no action needed.** Its pixel
  values are ~10,000x larger than the other two sites (band-0 median ≈179 against ≈0.018), uniformly
  across all 127 tiles. Cause: Pléiades reflectance products are delivered "in normalised reflectance
  values with a 1/10,000 ratio" (Airbus); two sites were divided through to true reflectance and
  `ireland2` was not. Confirmed by arithmetic — dividing `ireland2`'s per-band p98 by 10,000 gives
  0.0240 / 0.0251 / 0.0211 / **0.1156**, against `ireland1`'s 0.0285 / 0.0299 / 0.0249 / **0.1167**.
  Same product, same physical quantity, one missing division. The per-site stretch normalises it away
  entirely, so **this is NOT a Test B domain gap** and needs no more than an optional footnote.
  (The per-tile stretch had hidden it completely — every tile was rescaled to 0-1 individually, so
  nothing downstream could have surfaced it.)
- [ ] **Methods sentence for the normalisation change** (§2, one sentence, no discussion needed):
  percentiles are computed once per site rather than per tile — from training tiles for the inland
  site, and from their own imagery for the two held-out upland sites, which have no training tiles.
  OpenEarthMap tiles are already 8-bit and are not stretched. Built by
  `scripts/data_prep/build_normalisation_stats.py` -> `artifacts/normalisation_stats_<tag>.json`.

Why the change was made, in one line: the inherited per-tile stretch meant the same ground entered
the network under up to four different scalings (91-99.8% of shared pixels differ, up to 88 DN), and
**2.20% of shared foreground pixels changed class depending on which tile predicted them — 29x more
often within 0.5 m of a label boundary (15.21%) than in field interiors (0.52%)**. A model sits at
its decision threshold at a boundary and is confident in a field interior, so any input perturbation
flips predictions at boundaries and nowhere else. That is the same signature the paper attributes to
label ambiguity, from a different cause. Removing it removes the rival explanation; the boundary
analysis itself is unchanged.

## 7. Citation to add

Reina GA, Panchumarthy R, Thakur SP, Bastidas A, Bakas S (2020). Systematic Evaluation of Image
Tiling Adverse Effects on Deep Learning Semantic Segmentation. *Front. Neurosci.* 14:65.
doi:10.3389/fnins.2020.00065. Converted and read:
`references_md/reina-2020-systematic-evaluation-image-tiling-adverse-effects-deep-learning-2.md`.

Optional, for §2 if the subset choice needs a precedent: Cira C-I et al. (2024), *Remote Sensing*
16(16) 2954, doi:10.3390/rs16162954. Read pp. 1–2, 6–9. Not yet converted into `references_md/`.
