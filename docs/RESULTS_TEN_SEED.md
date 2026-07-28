# Results — ten seeds, campaign 650561, commit c5908c8

Computed 2026-07-28 from the completed campaign. **These supersede every five-seed figure in Parts 3
and 4.** Two of them change the narrative — see §6.

> **READING ORDER. §7 to §13 are the live numbers and supersede §1 to §6 wherever the two conflict.**
> §1-6 were written earlier the same day and three of their claims have since been retracted — §5 in
> full, the "near-symmetric" wording, and the interaction reading. Each retraction is marked where it
> occurs, but the sections were not deleted, because the retraction is part of the record.

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

> **RETRACTED. Do not use this section.** It compares absolute pixel counts between classes that
> differ in size by a factor of ten. Per pixel the flow runs TOWARD the majority class, roughly 8:1.
> §12 has the correct statement and the reason.

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

## 7. Paired contrasts with intervals

Recomputed from the raw per-seed values, not from the rounded SDs above. Paired t on ten seeds.
The interaction follows the Montgomery halving convention; the raw difference of differences is
twice the number shown, and the convention must be stated or a reader will mis-scale it.

**Test A**

| effect | mean | 95% CI | p | positive in |
|---|---|---|---|---|
| OpenEarthMap pre-training | -0.37 | [-3.67, +2.94] | 0.807 | 5/10 |
| class-balanced sampler | +0.19 | [-1.98, +2.36] | 0.847 | 5/10 |
| interaction | -2.08 | [-3.81, -0.36] | 0.023 | 1/10 |
| total (full - baseline) | -0.18 | [-3.09, +2.74] | 0.894 | 6/10 |

**The intervals are wider than the whole spread between the four cells** (0.6182 - 0.5937 = 2.45 pp).
Say so. The design cannot resolve an effect of +/-3 pp on aggregate mIoU, and a practitioner would
take a 3 pp gain. "Neither intervention moves the error" is not supported on the aggregate; what is
supported is a bound, and the per-class decomposition in §8.

## 8. Per-class paired contrasts — this is where the effects are

The aggregate is a mean over five classes. Decomposing it changes the finding.

**Test A** (pp of IoU, paired per seed)

| effect | Forest | Grassland | Cropland | Settlement | Seminatural |
|---|---|---|---|---|---|
| OEM pre-training | **+0.63** (8/10) | -0.60 (3/10) | -1.31 (5/10) | **+1.17** (9/10) | -1.73 (4/10) |
| sampler | +0.27 (7/10) | -0.29 (4/10) | +1.55 (6/10) | -0.15 (5/10) | -0.43 (4/10) |
| interaction | +0.00 (5/10) | -0.27 (3/10) | **-10.44** (2/10) | -0.14 (5/10) | +0.44 (4/10) |

**Test B** (pp of IoU, paired per seed)

| effect | Forest | Grassland | Cropland | Settlement | Seminatural |
|---|---|---|---|---|---|
| OEM pre-training | **+8.48** (9/10) | **-19.43** (0/10) | -2.74 (5/10) | **+11.32** (10/10) | -2.33 (3/10) |
| sampler | -4.86 (2/10) | +7.09 (8/10) | +0.02 (5/10) | -1.00 (4/10) | +7.49 (6/10) |
| interaction | +1.09 (6/10) | +5.71 (8/10) | +2.37 (7/10) | -0.04 (5/10) | +8.44 (8/10) |

Significant at p<0.05: Test A Settlement +1.17 (p=0.034) and the Cropland interaction -10.44
(p=0.020); Test B Forest +8.48 (p=0.003), Grassland -19.43 (p=0.003), Settlement +11.32 (p<0.001),
and three interaction terms.

**The pattern is the OpenEarthMap taxonomy mapping, class for class** (`geoseg/taxonomy.py:67-91`):

| our class | OEM classes mapped onto it | Test B effect of pre-training |
|---|---|---|
| Settlement | Developed 58%, Road 52%, Building 88% | **+11.32 pp, 10/10 seeds** |
| Forest | Tree 80% | **+8.48 pp, 9/10 seeds** |
| Grassland | Bareland 61%, Rangeland 57%, Agriculture 82% | **-19.43 pp, 0/10 seeds** |
| Cropland | *nothing* | -2.74, 5/10 — null |
| Seminatural | *nothing* | -2.33, 3/10 — null |

Three heterogeneous OEM classes collapse onto Grassland and Grassland gets worse in every seed. Two
of our classes receive nothing and nothing happens to them. The classes with clean, dedicated source
labels gain most. **Test A shows the same sign ordering at smaller magnitude** (Settlement and Forest
up, Grassland down, the other two null), so the pattern replicates on a second, independent landscape.

**This also disposes of the 2.00x gradient-step confound.** Factor A gives the transfer arm twice the
in-domain steps, so a reader may attribute any gain to extra training. Extra training predicts
improvement in every class. The observed pattern includes a 19.43 pp *loss* in 10 of 10 seeds. Extra
in-domain training does not do that.

**And it disposes of "neither intervention does anything".** Pre-training does a great deal. The
aggregate is near zero because a +11 and a -19 average out.

## 9. The interaction is a cropland effect

Test A interaction on mIoU is -2.08 pp; the Cropland term alone is -10.44/5 = -2.09 pp, i.e. **100%
of it**. Every other class is flat. Cropland is 1.35% of Test A foreground, in 52 of 294 tiles and 8
grid cells — the split's minimum acceptance floor — with a between-seed IoU SD of 0.195.

**So the interaction must be reported as a cropland effect with its support stated, or not at all.**
Reporting it as a general property of the two interventions, while §3 says no per-cell cropland claim
is supportable, is an internal contradiction a referee finds in one pass. The sign also reverses on
Test B (+3.51, 8/10), which is consistent with a landscape-specific cropland effect and not with a
property of the interventions.

## 10. The registered second arm — computed 2026-07-28, and it passes

Never run before today. The preregistration requires that across the four cells the **near-boundary
rate vary by less than the interior rate**, in relative terms, with the falsifier that if both move
proportionally the label-ceiling reading is unsupported.

Ten seeds, four cells, Test A, paired per seed. `n_near` and `n_far` are bit-identical across cells,
so the landscape is a constant in every comparison.

| band | spread across cells, near-boundary rate | spread across cells, interior rate | difference | seeds |
|---|---|---|---|---|
| 8 m | CV 3.53% | CV 14.10% | **+10.57 pts**, 95% CI [+8.26, +12.88], p<0.0001 | **10/10** |
| 1 m | CV 1.89% | CV 9.06% | **+7.17 pts**, 95% CI [+5.49, +8.85], p<0.0001 | **10/10** |

Range over mean tells the same story: at 8 m the near-boundary rate spans 7.85% of its mean across
the four cells, the interior rate 31.59%.

**THE SEED-ONLY CONTROL, computed 2026-07-28 after an adversarial check, and it changes the reading.**
The spreads above are taken across the four cells within a seed. Recomputing them across the ten
seeds within a cell — the same arithmetic with the roles swapped, so the interventions are held fixed
and only initialisation varies:

| band | | across the four CELLS | across the ten SEEDS |
|---|---|---|---|
| 8 m | near-boundary rate | CV 3.53% | **CV 3.53%** |
| 8 m | interior rate | CV 14.10% | **CV 13.51%** |
| 1 m | near-boundary rate | CV 1.89% | **CV 1.86%** |
| 1 m | interior rate | CV 9.06% | **CV 8.61%** |

**They are the same.** The variation across the four curation configurations is run-to-run seed
variance and nothing else. So the registered condition holds literally — the near-boundary rate does
vary less than the interior rate — but it holds for a reason that has nothing to do with the
interventions, and **the reading "the interventions move interior error but not boundary error" is
FALSE and must not be written.** They move neither.

**What is licensed instead.** The near-boundary error rate is a near-invariant of this dataset and
architecture: **19.49% at 8 m and 40.82% at 1 m, with a coefficient of variation of 3.5% and 1.9%
across all forty runs**, spanning four curation configurations and ten initialisations. The interior
rate is intrinsically the more variable of the two, run to run, by roughly a factor of four in
relative terms — a property of the estimator, not an effect of the interventions.

Absolute ranges across the four cells, for reporting beside the relative form: 1.53 pp near against
2.85 pp interior at 8 m; 1.74 pp against 2.20 pp at 1 m. The relative form is what was registered, and
it is measured against a near-boundary rate roughly twice the interior rate, so it overstates the
contrast if quoted alone.

**The registered falsifier did not fire** — the two rates do not fall proportionally, because neither
falls. But the arm is closer to uninformative than to passed, and it must be reported that way, with
the seed control printed beside it.

**Reproduced on the deduplicated scoring subset, 2026-07-28 (Slurm 652577), so this no longer rests
on an ad hoc run.** 160 runs of `boundary_rate_ratio.py`, one per cell x seed x band x tile
population, through `narrative_numbers.py`:

| band | | across the four CELLS | across the ten SEEDS |
|---|---|---|---|
| 8 m | near-boundary rate | CV 3.76% | **CV 3.70%** |
| 8 m | interior rate | CV 13.89% | **CV 14.21%** |
| 1 m | near-boundary rate | CV 1.94% | **CV 1.89%** |
| 1 m | interior rate | CV 9.17% | **CV 9.31%** |

Same conclusion on the 90-tile deduplicated population as on all 294 tiles. The cell spread is seed
noise on either population.

**rho on the deduplicated subset, per seed, ten seeds:**

| cell | rho at 1 m | rho at 8 m |
|---|---|---|
| baseline | 3.692 +/- 0.329 | **2.137** +/- 0.304 |
| transfer only | 3.624 +/- 0.512 | 2.062 +/- 0.405 |
| sampler only | 3.681 +/- 0.308 | 2.076 +/- 0.294 |
| full | 3.436 +/- 0.238 | 1.903 +/- 0.168 |

Against 2.28 at 8 m on all 294 tiles. **rho and the reported IoU are now censuses of the same 90
tiles**, which resolves the population asymmetry §12 flags. Report the deduplicated figure and say
which population it is.

**Two things this is not.** The interior rate does not *fall* — the full model has the highest
interior rate of the four (9.69% against baseline 8.78%). The registered wording is about
variability, and that is what is claimed. And it is a **necessary condition, not a diagnosis**: every
rival cause that is constant across the four cells — encoder-decoder edge blur, mixed pixels at
0.5 m, registration offset — predicts the same flat near-boundary rate, and the architecture is
constant in every contrast this design computes.

## 11. rho on the clean split, ten seeds, four cells, Test A

Per seed, not from the ensemble. The ensemble argmax inflates the ratio by removing interior error
preferentially, which is the mechanism that disqualified the retracted `lift` statistic.

| cell | rho at 1 m | rho at 8 m |
|---|---|---|
| baseline | 3.85 +/- 0.37 | 2.28 +/- 0.35 |
| transfer only | 3.75 +/- 0.50 | 2.18 +/- 0.43 |
| sampler only | 3.82 +/- 0.26 | 2.22 +/- 0.28 |
| full | 3.57 +/- 0.23 | 2.03 +/- 0.16 |

Underlying rates, baseline: 41.18% within 1 m against 10.78% beyond; 19.63% within 8 m against 8.78%
beyond. **The ratio rises as the band narrows** — that shape is the evidence, and a single width
would hide it.

Volpi & Tuia's 1.24-1.33 is **not** a comparator: their band is 3 px at 9 cm, 30-53x narrower.

## 12. Confusion structure and the imbalance question, restated correctly

Ten seeds pooled, Test A baseline, deduplicated scoring subset. 208.10M foreground pixels,
27.63M errors.

**§5 above is wrong and must not be written.** "The flow runs against the majority class" was
computed from absolute pixel counts between classes that differ in size by 9.8x. Per pixel:

| | rate |
|---|---|
| semi-natural pixels called grassland | **39.2%** |
| grassland pixels called semi-natural | **4.8%** |

That is the textbook majority-class signature, 8:1 toward the majority. Absolute counts cannot test a
directional claim about prevalence, and PT3 §2 states the lesson in one direction while §5 applies it
backwards in the other.

**What replaces it — three things, all defensible.**

**(a) Predicted area does not track training frequency.** Baseline predicted against reference area:

| class | train share | predicted vs reference area |
|---|---|---|
| Settlement | 3.34% (rarest) | **+5.1%** |
| Seminatural | 4.26% | **+11.4%** |
| Cropland | 7.70% | **-45.6%** |
| Forest | 14.58% | +3.6% |
| Grassland | 70.11% | -1.3% |

The two rarest classes are predicted *over* their true extent. The one badly under-predicted class is
nearly twice as common in training as either. Under-prediction is not a function of rarity here.

**(b) The sampler acted, and bought almost nothing.** Class-balanced sampling shifts the shipped
model exactly as designed: predicted semi-natural rises from 16.23M pixels to 18.02M against a
reference 14.57M, i.e. **24% more semi-natural than exists**. Semi-natural recall moves 56.03% to
**56.40%**, and precision falls 50.3% to 45.6%. **Of every 100 extra pixels the sampler pushed into
semi-natural, 3 were right.** The confusion into grassland does not shift either: 39.15% to 39.80%.

This is the strongest form of the null. The intervention is not inert — it moves the decision
boundary as intended — and the classification does not improve. The model is not reluctant to say
semi-natural; it cannot tell which pixels are semi-natural.

**(c) WITHDRAWN 2026-07-28, same day, after an adversarial check.** An earlier version of this
section claimed that per-class accuracy is ordered by the number of distinct places a class occupies
rather than by its pixel share, with Spearman +0.600 / +0.900 / +0.975 over pixel share, training
tiles and 950 m blocks, and used it to explain the sampler null. **Do not write any of that.** It
failed on four counts:

- **The p-values were wrong.** They came from SciPy's asymptotic approximation, which is meaningless
  at n=5. Exact two-sided permutation p-values over all 120 orderings are 0.350, **0.083** and
  **0.033**, not 0.285, 0.037 and 0.005. The tile predictor is not significant. Selecting the best of
  three predictors, the chance of reaching rho>=0.90 under the null is 1 in 12.
- **The three predictors are one variable.** Tiles and blocks rank-correlate at +0.975 and are
  rank-identical apart from one tie. Train holds exactly 32 blocks and Forest and Grassland both sit
  at 32, so the block variable is censored at its ceiling on the one pair the tile ordering gets
  wrong. The +0.900 to +0.975 gap is that saturation tie, not information.
- **It reduces to a single pairwise comparison.** Dropping any one of Cropland, Settlement or
  Seminatural makes pixel share and tile count exactly equally good. The whole claim is: settlement
  has 3.34% of training pixels and scores 0.711, cropland has 7.70% and scores 0.340.
- **It is confounded with test geometry.** The same IoU ordering is reproduced perfectly
  (Spearman +1.000) by the number of *test* tiles containing each class, and by the number of
  connected components in the test masks. Neither involves training exposure at all.

**And the inference did not follow even if the ordering held.** Variation across five classes at one
fixed configuration cannot license a counterfactual about what happens to one class when its sampling
rate changes. Place count is manipulated in no arm of the design. And there is a direct
counterexample in §8: on Test B the sampler moves Seminatural **+7.49 pp** in 6 of 10 seeds, which a
training-set mechanism must predict a null for on both test sets.

**What survives, and it is only this.** Two facts about the interventions, both verifiable from code
and neither statistical:

- The class-balanced sampler draws with replacement from the same 1,072 training tiles
  (`WeightedRandomSampler`, `num_samples=len(train_dataset)`), so it cannot introduce ground the
  training set does not already hold. The weights file holds three values over 261 / 550 / 261 tiles
  and reproduces the documented 1.27x settlement and 2.84x semi-natural uplift.
- No OpenEarthMap class maps to Cropland or to Seminatural (`geoseg/taxonomy.py:67-91`), so
  pre-training cannot supply labelled examples of either. **State it that way** — OEM does add 2,118
  new scenes overall, so "adds no places" is false; what it adds no places of is these two classes.
  Any effect on them is representation transfer, not label transfer.

Report §12(b) — the sampler acted and bought almost nothing — as an observation. Do not attach an
explanation of why to it.

**Class pair shares are unchanged** (recomputed, matching §4): Grassland->Seminatural 7,195,002 px and
Seminatural->Grassland 5,704,254 px, together **46.68%** of all foreground error.

## 13. Still outstanding

Nothing computational. What remains is the manuscript.
