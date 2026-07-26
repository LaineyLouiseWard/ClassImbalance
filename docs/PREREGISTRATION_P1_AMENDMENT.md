# WITHDRAWN 2026-07-26 — before any model was trained on the corrected split

This pre-registration is withdrawn. No model had been trained on the spatially blocked split at the
time of withdrawal, so no result influenced the decision.

**Reason.** Pre-registration is not standard practice in this field; it was adopted as credibility
armour after the first campaign was withdrawn for leakage. It proved to cost more than it bought: the
statistic required a threshold, the threshold required a provenance, and the only calibration data
available came from the withdrawn campaign. Each layer added a number that could be argued with, on top
of evidence — the trimap exclusion curve — that already showed the effect directly and is established
for exactly this purpose (Kohli et al. 2009; Csurka et al. 2013).

**What replaces it.** The boundary claim is supported descriptively: the trimap exclusion curve, and
boundary-band versus deep-interior error rates per class, with per-class support stated beside every
number. No threshold, and nothing "fails".

The full history below is retained unchanged, because a timestamped commitment should be superseded in
the open rather than deleted. See `docs/DECISIONS_REBUILD_2026-07.md`, D16.

---

# Pre-registration §P1 — amendment history and current registered form

**Status: all changes below were made BEFORE any model was trained on the spatially blocked split.**
No accuracy, error share or boundary statistic from that design exists at the time of writing. Two
superseded versions are preserved in full, because a pre-registration whose history is not auditable
is not a pre-registration.

---

## Version 0 — original §P1, verbatim (SUPERSEDED)

> The paper's claim is that residual error is **dominated** by boundary ambiguity. The statistic for
> that is the **share of total error falling within 8 m of a boundary** — not the raw boundary and
> interior rates, which are vulnerable to floor and ceiling effects.
>
> **Pre-registered:** the label-ceiling claim holds if the near-boundary error share stays above
> **65% on both test sets**. If it falls to ~40%, the claim is dead regardless of how large the ratio
> between boundary and interior rates looks.

---

## Version 1 — "boundary enrichment ratio E ≥ 2.0" (RETRACTED 2026-07-26, same day)

Registered and retracted within hours, before any training. Retained in full because it was committed
to this repository and must not disappear.

Version 1 replaced the absolute share with E = (error share within 8 m) / (foreground area share
within 8 m), registering denominators of 38.0% (Test A) and 22.2% (Test B) and a threshold of E ≥ 2.0
on both, dead below 1.5.

## Why version 1 was retracted

**It is not scale-free, which was its entire justification.** E's maximum attainable value is 1/p,
where p is the band's area share: 2.65 on Test A, 4.51 on Test B. A common threshold therefore sits at
different points in the two attainable ranges. Expressed as what it demands of the underlying
mechanism — the rate ratio ρ between near-boundary and interior error — E ≥ 2.0 requires ρ = 5.09 on
Test A and ρ = 2.80 on Test B, a factor of 1.82. Version 0 required ρ = 3.06 and 6.51, a factor of
2.13 in the other direction. **Version 1 moved the incomparability, reversed its sign, and barely
shrank it.**

**Its justification for retiring version 0 was itself the error it claimed to be correcting.** The
argument was that 65% "has never once been reached on what Test B would require (2.93x)". That
compares an enrichment requirement at p = 0.222 against enrichments measured at p = 0.346 and 0.314 —
an invalid cross-p comparison by version 1's own logic. Translating each prior measurement through its
ρ, the full-model test measurement (ρ = 12.02) predicts a 77.4% error share on Test B, comfortably
above 65%. **The claim that version 0 was unreachable on Test B was not established.**

**Its stated safety margin was an order of magnitude smaller than the uncertainty.** Version 1 recorded
a 3.5% margin to failure. A tile-level bootstrap over the 44 validation tiles holding unseen
foreground gives E = 2.105, 95% CI [1.494, 2.907], P(E < 2.0) = 0.379; four of ten per-seed values fall
below 2.0; and re-expressed at Test A's actual p the weakest prior lands at 1.968, already failing.

**Its registered denominators were unreproducible and internally inconsistent.** No script in the
repository produced 38.0% or 22.2%. Recomputed: Test A is 37.774% under strict `< 8.0` and 38.033%
under `<= 8.0`; Test B is 22.199% under both. The two registered numbers used *different* band
conventions, while the pipeline computing the numerator uses strict `<`.

**It could not adjudicate the claim it was written for.** On identical pixels the baseline model gives
E = 1.828 and the full model E = 2.070: a 16.7% error reduction bought a 13.2% rise in E. E rises as a
model improves, because improving a model removes interior error first. A threshold on E is therefore
partly a test of how well the model was trained — which is the alternative hypothesis the label-ceiling
claim is meant to exclude.

Version 1's form was not novel: E is exactly **lift** (Brin et al. 1997, "interest"), standard in
association-rule mining. That literature documents this precise defect. Shaikh, McNicholas, Antonie &
Murphy (2013, arXiv:1308.3740) §2.2 give two rules with identical lift 1.95 whose maxima are 2 and 10,
concluding *"the interpretation of the two lift values should not be the same because the maximum
attainable value is different."* Substituting "within 8 m of a boundary" reproduces the Test A / Test B
situation verbatim.

---

## Version 2 — CURRENT REGISTERED FORM

## The statistic

> **ρ = (foreground error rate within 8 m of a GT boundary) / (foreground error rate beyond 8 m)**
>
> Both are rates over pixel counts, so ρ has no area denominator and its numerical value does not
> depend on how finely the landscape is parcelled. It is the only quantity in this family with that
> property: the error *share* within the band, and any ratio of that share to the area share, are both
> explicit functions of the band's area fraction at fixed ρ.
>
> Error is the ten-seed ensemble argmax against GT, foreground pixels only (`ignore_index = 0`).
> Distances are per-site in metres via `sampling=(gsd_y, gsd_x)`: inland 0.500 × 0.500 m, ireland1
> 0.515 × 0.641 m, ireland2 0.515 × 0.634 m. Band membership is strict, `distance < 8.0 m`.

## The threshold

> **Pre-registered:** the boundary-concentration claim holds if **ρ ≥ 4.0 on both test sets**, judged
> on the **lower bound of a tile-level bootstrap 95% CI**, resampling by non-overlapping footprint
> group rather than by tile id. The claim is **dead below ρ = 2.0**; between 2.0 and 4.0 it is reported
> as weak and the framing drops from "dominated by" to "concentrated at".

Grounding, from the three existing measurements on genuinely unseen pixels: ρ = 3.254 (baseline model,
validation), 4.771 (full model, validation), 12.018 (full model, test). A threshold of 4.0 sits 16%
below the weakest full-model estimate and *above* the baseline model, so the baseline fails it and the
full model is genuinely at risk. Requiring the bootstrap lower bound rather than the point estimate is
what makes it bind.

## The second arm — because ρ locates error but does not diagnose its cause

ρ rises as a model improves, so a high ρ alone is consistent with both "labels are the binding
constraint" and "the model is simply well trained and has cleaned up the easy interior first". The
registered claim therefore requires a second, discriminating observation across the four factorial
cells, which costs nothing extra because all four are trained anyway:

> **Registered:** if the residual error is label-limited at boundaries, the **near-boundary error rate
> must be approximately flat across the four cells** while the **beyond-8 m rate falls**. Specifically,
> across baseline → transfer → sampler → full, the near-boundary rate must vary by less than the
> interior rate does, in relative terms. If both fall proportionally, the concentration is a property
> of model quality and the label-ceiling interpretation is not supported by this evidence.

## What must be reported alongside, and is binding

1. **`a`, `p`, `E`, the ceiling `1/p`, and `ρ`, for each test set and for ireland1 and ireland2
   separately.** Test B is itself a pool of two sites whose band area shares are 38.89% and 15.33% and
   whose ceilings differ by 2.5×; ireland2 supplies 71% of Test B's foreground. Pooling them hides a
   sign reversal already observed — ireland1 is *more* boundary-dense than the training distribution
   and behaves like Test A.
2. **The per-class table with class-mix weights.** Per-class ceilings span 13× (Test A forest 1.26,
   Test B semi-natural 15.88), so a pooled pass/fail is largely a statement about class mix.
   Semi-natural is 60.3% of Test B foreground and is the one class that showed *no* boundary
   concentration on validation unseen pixels.
3. **Treatment of boundary-free tiles.** 19 of 191 Test B tiles contain no GT boundary at all, holding
   16.17% of its foreground; ρ is undefined there. They are **excluded** from both numerator and
   denominator, which moves Test B's band area share from 22.199% to **26.480%**. Test A has none.
4. **The denominators as a committed artefact**, produced by a script in this repository, strict
   `< 8.0`: Test A **37.774%**, Test B **26.480%** after the exclusion above. A number no script can
   reproduce is not registered.
5. **Test B is unprecedented, not extrapolated.** The prior 3.25–12.02 range for ρ is 97–99.9% inland:
   the entire upland unseen-pixel evidence is 3 tiles, 79,262 pixels and 379 errors, with the error
   share inside the band equal to 1.000 on both sites. The Test B threshold is a genuine prediction.

## Sensitivity, recorded now so it cannot be chosen later

The 8 m band predates this amendment and is frozen. It is not neutral: on validation, E crosses 2.0 at
about 8.7 m, and the band sweep gives 6 m → 2.375, 8 m → 2.070, 12 m → 1.741, 16 m → 1.558. ρ is less
band-sensitive than E but not immune. 8 m at 0.5 m GSD is 16 px, matching Kohli, Ladický & Torr (2009),
whose trimap panels use 8- and 16-pixel bands; Cheng et al. (2021) recommend setting the width from
annotation consistency, which this dataset cannot supply (see below).

## What does not change

- Both test sets must clear the threshold. Test B is not exempted.
- The unseen-pixel baseline remains the reference, never the all-pixel figure.
- The manuscript's **92% / 96%** near-boundary shares are leakage-inflated all-pixel figures and must
  not appear. The unseen-pixel equivalents are 71.6% / 84.6%.

## Related limitation, established by measurement

The dataset carries a single annotation pass: 5,898,240 co-labelled pixels across 60 overlapping tile
pairs are 100.0000% identical, so the labels are one vector delineation rasterised into tiles. No
inter-annotator agreement figure can be recovered, and none can be produced without re-annotation. The
paper therefore cannot separate annotator inconsistency from vector-to-raster quantisation from genuine
mixed pixels. All three are limits on boundary *delineation*, which is why the claim is scoped to
boundary delineation rather than to label quality in general.

---

## Provenance

Version 1 was drafted, attacked by two independent adversarial reviews, and retracted the same day. The
literature review established that its form was lift and that lift's own literature documents its
defect. The adversarial statistical review reproduced all of version 1's numbers to four decimals,
recorded 11 of its own attacks as failures, and supplied both the ρ formulation and the objection that
motivates the second arm. Both reviews are in `notes/`.
