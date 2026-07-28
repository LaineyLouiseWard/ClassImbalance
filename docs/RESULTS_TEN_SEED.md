# Results — ten seeds, campaign 650561, commit c5908c8

Computed 2026-07-28 from the completed campaign. **These supersede every five-seed figure in Parts 3
and 4.** Two of them change the narrative — see §6.

## What these numbers are, and are not

- **Deduplicated.** Scoring used the non-overlapping subset: 20.81M foreground pixels on Test A
  against 69.56M in the full 294-tile set, i.e. 29.9%, matching 90 of 294 tiles. Every patch of ground
  is counted once.
- **Test A** = the inland held-out strip, 1,664 m from any training ground. **This is the reporting
  surface.**
- **Test B** = the two upland sites, held out whole. A different landscape, not a replicate.
- **Validation is listed once, for completeness, and must not be reported.** It sits 256 m from
  training and every checkpoint in every cell was selected on it. Note it is also the only split where
  the cells look ordered the way the old campaign claimed - that is the selection effect, not a result.
- No test-time augmentation. Background excluded from every mIoU.

## 1. Foreground mIoU, mean +/- SD over ten seeds

| split | baseline | transfer only | sampler only | full |
|---|---|---|---|---|
| **Test A** | 0.5954 +/- 0.027 | 0.6126 +/- 0.038 | 0.6182 +/- 0.039 | 0.5937 +/- 0.036 |
| **Test B** | 0.4883 +/- 0.058 | 0.4438 +/- 0.055 | 0.4707 +/- 0.060 | 0.4964 +/- 0.073 |
| val *(not for report)* | 0.6644 +/- 0.009 | 0.6597 +/- 0.015 | 0.6506 +/- 0.010 | 0.6559 +/- 0.017 |

## 2. Paired per-seed factorial contrasts (percentage points)

**Test A**

| effect | mean | SD | positive in |
|---|---|---|---|
| OpenEarthMap pre-training | **-0.37** | 4.62 | 5/10 |
| class-balanced sampler | **+0.19** | 3.03 | 5/10 |
| interaction | **-2.08** | 2.42 | **1/10** |
| total (full - baseline) | -0.18 | 4.08 | 6/10 |

**Test B**

| effect | mean | SD | positive in |
|---|---|---|---|
| OpenEarthMap pre-training | -0.94 | 6.58 | 3/10 |
| class-balanced sampler | +1.75 | 4.01 | 6/10 |
| interaction | +3.51 | 4.91 | 8/10 |
| total | +0.81 | 7.75 | 4/10 |

**Read this carefully.** Both main effects are indistinguishable from zero with the sign splitting
5/10. The only contrast with a consistent sign anywhere is the **Test A interaction, negative in 9 of
10 seeds** - the two interventions together are worse than either alone.

## 3. Per-class IoU, Test A, ten seeds

| class | baseline | transfer only | sampler only | full |
|---|---|---|---|---|
| Forest | 0.721 +/- 0.007 | 0.727 +/- 0.005 | 0.723 +/- 0.008 | 0.730 +/- 0.005 |
| Grassland | 0.846 +/- 0.011 | 0.843 +/- 0.018 | 0.846 +/- 0.012 | 0.837 +/- 0.010 |
| **Cropland** | 0.340 +/- 0.107 | 0.431 +/- 0.133 | 0.460 +/- 0.166 | 0.342 +/- 0.195 |
| Settlement | 0.711 +/- 0.014 | 0.724 +/- 0.008 | 0.711 +/- 0.021 | 0.721 +/- 0.008 |
| **Seminatural** | 0.360 +/- 0.057 | 0.338 +/- 0.074 | 0.351 +/- 0.044 | 0.338 +/- 0.035 |

Cropland's SD reaches 0.195 - over half its own mean. **No per-cell cropland claim is supportable.**
The three strong classes vary by about a point; the two weak ones swing wildly.

## 4. Confusion structure, Test A, baseline, ten seeds pooled

Absolute pixels. 208.10M foreground pixels scored, **27.63M foreground errors**.

| flow | share of all error | pixels |
|---|---|---|
| Grassland -> Seminatural | **26.04%** | 7.20M |
| Seminatural -> Grassland | **20.64%** | 5.70M |
| Grassland -> Forest | 17.73% | 4.90M |
| Forest -> Grassland | 12.33% | 3.41M |
| Cropland -> Grassland | 5.90% | 1.63M |
| Grassland -> Settlement | 3.27% | 0.90M |

**The grassland pair is 46.68% of all foreground error.** Forest and grassland together are 30.06%.

**Net flow into Grassland** (positive = grassland gains pixels):

| | net |
|---|---|
| Seminatural | **-1,491,076** |
| Forest | -1,490,178 |
| Settlement | -199,570 |
| Cropland | **+1,259,903** |

## 5. The asymmetry, stated precisely

Grassland -> Seminatural exceeds the reverse by a **ratio of 1.261**, a net 1.49M pixels. On five
seeds the ratio was 1.17; ten seeds made it slightly larger, not smaller.

**So "near-symmetric" is no longer accurate and must not be written.** What survives, and it is the
part that carries the argument: **the flow runs AGAINST the majority class.** Grassland is 70% of
training pixels and is a net donor to semi-natural, to forest and to settlement. Class imbalance
predicts the opposite direction. The only class grassland absorbs on net is cropland.

## 6. What changed from the five-seed narrative

**(a) OpenEarthMap pre-training does NOT give +2.6 pp.** That was a five-seed artefact from seeds
42-46. On ten seeds it is **-0.37 pp, positive in 5 of 10**. Any sentence crediting transfer with a
gain must go, including the plan's R1 "upper bound" framing, which assumed a positive effect to bound.

**(b) "Near-symmetric" becomes "runs against the majority".** 1.26:1 is a 26% excess, not symmetry.
The conclusion is unchanged and arguably strengthened - imbalance cannot explain a flow that runs from
the majority class to the minority one - but the wording must change.

**(c) Unchanged:** the pair carries about half the error (46.68%, was 49%); both interventions are
small; the weak classes are cropland and semi-natural.

**(d) New, and the only consistent contrast in the study:** the Test A interaction is **negative in 9
of 10 seeds**. Applying both interventions together is worse than either alone. Report it; do not yet
interpret it.

## 7. Still outstanding

The boundary sweep on ten seeds, four cells, both test sets - currently one seed, baseline, Test A
only. And the across-cell arm, never computed.
