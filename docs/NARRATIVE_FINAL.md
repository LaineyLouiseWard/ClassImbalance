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
| 6 | **19.5%, CV 3.5%** | near-boundary error rate across all forty runs |

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

**The most-cited signature is present.** The two weak classes are minorities against a grassland
majority holding 70% of training pixels, and their errors run toward that majority: **semi-natural
grassland is predicted as grassland 39% of the time, grassland as semi-natural 5%.** State the
qualification in the same breath — grassland outnumbers semi-natural about 10:1 on the scored pixels,
so that asymmetry is no larger than prevalence alone predicts. It is consistent with imbalance; it is
not evidence of a bias beyond prevalence.

**So we tested it, with one asymmetry declared up front.** Class-balanced sampling reweights the
examples we hold. OpenEarthMap pre-training adds examples — but no OpenEarthMap class maps to cropland
or to semi-natural grassland (`geoseg/taxonomy.py:67-91`), so for those two classes it is not an
imbalance remedy and cannot be read as one. **The imbalance test for the two weak classes rests on the
sampler.** Pre-training is in the design as the other lever a practitioner would pull, and its effect
on those two classes is representation transfer, not label transfer. Both crossed in a 2x2 over ten
seeds, architecture fixed, contrasts paired within each seed.

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
with each other, in both directions, is **47% of all foreground error** on Test A at baseline. The next
largest pair, forest and grassland, is about 30% — so say "the largest pair", never "no other pair is
close".

**Weight each pair by the area it could occupy, because grassland covers most of the scene and any
pair containing it starts large.** On that basis the grassland pair runs at **2.1 times** what area
predicts and forest-grassland at 0.6 times. That is the statistic that carries the claim.

A per-class table hides this, because it is a property of a pair. Read the confusion matrix on
absolute pixel volumes as well as rates — row percentages make a small class look dominant, absolute
counts make a large one look like a donor, and either alone misleads.

## Q3 — Where is the error, in space?

**At the lines between parcels.** The foreground error rate within 1 m of a reference boundary is
**3.85 times** the rate beyond it, and 2.28 times at 8 m.

**Report it at more than one width, not at one.** The ratio rises as the band narrows. Where a single
width has been justified in the literature it has been justified by annotation consistency across
repeat passes (Cheng et al. 2021), which a single-pass dataset cannot supply.

## Q4 — How stable is that boundary error?

**Very.** The foreground error rate within 8 m of a reference boundary is **19.49%, coefficient of
variation 3.5%, across all forty runs** — four curation configurations and ten initialisations. At
1 m it is 40.82%, CV 1.9%.

**This is a stability result and nothing more.** Holding the cell fixed and varying only the seed
gives the same spreads as varying the cell, so the across-cell comparison carries no information
about the interventions. The registered falsifier did not fire, because neither rate falls. Report
the arm as uninformative, with the seed control printed beside it.

**What it does not license.** Not that curation moves interior error but not boundary error — it
moves neither. Not that the labels are the cause. The paper locates the error and stops.

---

## The take-home

**The residual error is one class pair, concentrated where the classes meet, at a rate that does not
respond to either standard curation lever within the +/-3 percentage points this design can resolve.**

Before spending on more data or on rebalancing, measure where your errors sit relative to your class
boundaries. That measurement costs nothing and it is available before the money is spent.

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
