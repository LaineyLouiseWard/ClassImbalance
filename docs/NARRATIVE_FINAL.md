# The narrative

Rewritten 2026-07-28 on the four-question spine. This replaces the earlier version of this file,
which led on per-class pre-training effects — those turned out to rest on Test B, which is not the
reporting surface, and are demoted to support.

**Six numbers carry the whole paper.** Nothing else needs to be in the abstract, the introduction or
the conclusions. Everything below is anchored to them and to nothing else.

| | number | what it is |
|---|---|---|
| 1 | **39% against 5%** | semi-natural grassland predicted as grassland, against the reverse |
| 2 | **±3 pp** | the most either intervention could have moved foreground mIoU |
| 3 | **24%, and 3 in 100** | how much extra semi-natural the sampler claims, and how much of it is right |
| 4 | **47%** | share of foreground error that is the grassland pair, both directions |
| 5 | **3.85** | error rate within 1 m of a boundary over the rate beyond it |
| 6 | **10 of 10** | seeds in which the registered across-cell condition holds |

---

## Terminology, fixed

Use these words and no others. Several are forbidden alternatives, not stylistic preferences.

| write | never write | why |
|---|---|---|
| OpenEarthMap pre-training | cross-dataset transfer *(as the name of factor A)* | the arm gets 2.00x the in-domain gradient steps, so the effect cannot be attributed to transfer as a mechanism (`DO_NOT_ADD.md`) |
| class-balanced sampling (clsbal) | rebalancing, resampling | the shipped method has a name |
| foreground mIoU | accuracy, mIoU | background is excluded from every reported metric |
| Test A, Test B | the test set, held-out data | there are two and they are never pooled |
| reference labels, ground-truth boundary | truth, the truth | the reference is not truth |
| semi-natural grassland, grassland | the grasslands, rough grazing *(in the paper)* | these are class names; plain synonyms belong in the cover letter, not the manuscript |
| paired per-seed contrast | improvement, gain | contrasts carry intervals; levels do not |
| *(describe the disjunction)* | label-quality ceiling, diagnosing | forbidden claim, and the paper declines to diagnose |
| there is no line on the ground | gradational, grades into | jargon |

---

## Q1 — Is this an imbalance problem?

**Everything observable says yes, before any intervention.** The two weak classes are minorities
against a grassland majority that holds 70% of training pixels, and their errors run toward that
majority: **semi-natural grassland is predicted as grassland 39% of the time, and grassland is
predicted as semi-natural 5%.** That asymmetry is what class imbalance predicts.

**So we tested it.** The two standard responses are to add examples or to reweight the examples you
have. OpenEarthMap pre-training and class-balanced sampling, crossed in a 2x2 over ten seeds, on a
fixed architecture, with contrasts paired within each seed.

**Neither moves foreground mIoU. The design can exclude a gain larger than about 3 percentage
points**, and neither comes close to it.

**And the sampler is not inert — this is the part that matters.** It does exactly what it is built to
do: the model goes on to claim **24% more semi-natural grassland than the reference holds**. Of every
hundred extra pixels it claims, **three are right**. Recall moves by less than half a point.

> **The claim, stated as it must be:** the standard signature of class imbalance is present, and
> imbalance-targeted curation does not shift it. Not "imbalance is not the cause", and not "the
> sampler has no effect" — both are forbidden, and neither is what was measured.

**Why this belongs in the paper at all.** Without it, every later claim about where the error sits is
unlicensed, because the obvious explanation has not been ruled out. The factorial is the control that
makes the rest of the paper possible. It is not the headline.

**One thing not to claim.** Rarity does not order difficulty here — settlement is the rarest class in
training and among the best segmented, while cropland has more than twice its pixels and is the worst.
Five classes cannot support a claim in either direction. State the ordering if it is useful; fit
nothing to it.

## Q2 — Where is the error, by class?

**Not spread across classes. Concentrated in one pair.** Grassland and semi-natural grassland confused
with each other, in both directions, is **47% of all foreground error**. No other pair is close.

This is invisible in a per-class table, because it is a property of a pair rather than of a class.
Read the confusion matrix on absolute pixel volumes, never row-normalised — row percentages make a
small class look dominant, and absolute counts make a large one look like a donor. Both mislead on
their own.

## Q3 — Where is the error, in space?

**At the lines between parcels.** The foreground error rate within 1 m of a reference boundary is
**3.85 times** the rate beyond it, and 2.28 times at 8 m.

**Report it as a curve across band widths, not at one width.** The ratio rises as the band narrows,
and that shape is the evidence. No single width can be justified, and the aerial benchmarks that
erode a boundary band away before scoring never justify theirs either.

## Q4 — Are those boundaries a property of the labels or of a weak model?

**One model cannot answer this**, because the ratio rises as any model improves — a better model
removes interior error first.

**So compare models.** Across the four factorial cells, the error rate away from boundaries varies
substantially more than the error rate at boundaries does, in **10 of 10 seeds**, at both band widths.

**What that implies, spelled out.** If the error at boundaries were simply the hardest part of a job
the model is not yet good enough at, then changing the model's training data would move it, along with
everything else. The four cells are not four copies of one model — the interior error rate genuinely
responds to which cell you train. So the interventions do reach the model. They just do not reach the
boundaries.

**The licensed conclusion:** the error at class boundaries is not something data curation moves. It is
insensitive to the two levers that demonstrably move error elsewhere in the same images. What remains
as its cause is the reference labels, or the architecture — and this design cannot separate those two,
because the architecture is held constant in every contrast it computes. That is the open disjunction
in the Limitations, and it is why the paper locates the error rather than diagnosing it.

**Three things that travel with it, always.**

- It was **registered before any model was trained**, with a stated falsifier, and this is the first
  time it has been computed.
- It is a **necessary condition, not a diagnosis.** Anything that is constant across the four cells —
  including the architecture's own tendency to blur edges — predicts the same flat boundary rate, and
  the architecture is constant in every contrast this design computes.
- Report absolute spreads beside the relative ones. The relative form is what was registered, but it
  is measured against a near-boundary rate that is roughly twice the interior rate, so the relative
  number overstates the effect if quoted alone.

---

## The take-home

**The error looks like an imbalance problem by every measure available without intervening, and
imbalance-targeted curation does not move it. What is left is one class pair, concentrated where the
classes meet.**

## What the paper is

A worked diagnosis, demonstrated end to end on operational imagery. The four questions above are the
contribution, in that order. The company's dataset is the case; the reader is a practitioner facing
the same situation.

The contribution is **quantification and procedure, not a new concept**. Say so in one sentence in the
Introduction. Taxonomy alignment being lossy, and boundary error being hard, are both already cited on
the paper's own second page. Claiming discovery invites a one-line rejection quoting the manuscript
against itself.

## What stays open, and is said so

Whether the two grassland classes are inseparable in this imagery or were labelled inconsistently
cannot be settled here. It needs a second independent annotation pass over the same ground, which does
not exist. **Stated as an open question in Limitations, never as a diagnosis** — and for a transition
with no line on the ground the two are not fully separable even in principle.

Scope: one annotator, one protocol, one acquisition year, training data from a single site. What is
characterised is the ceiling imposed by *that* annotation process, not a property of dense land-cover
annotation in general.

## Demoted, and why

**Per-class pre-training effects.** On Test B they are large — settlement and forest gain, grassland
loses heavily — and they follow the OpenEarthMap mapping. But Test B is two purposively chosen upland
sites and cannot carry a headline. On Test A, the reporting surface, the same effects are small.
Report them as a secondary observation with a warning that per-class effects differ in sign and are
hidden by an aggregate score. **Do not claim the mapping predicts which way a class moves** — it
licenses only whether a class is reachable at all, and mapping structure is confounded with class
prevalence across five classes.

**The mapping's provenance, stated correctly.** It is the argmax of a trained OpenEarthMap teacher's
confusion on the training split, regrounded 2026-07-26. Write "fixed before any of the four factorial
cells was trained, and derived only from the training split". Never "before any model existed".
