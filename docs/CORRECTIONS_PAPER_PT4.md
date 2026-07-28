# Part 4 — the narrative we are going with

Parts 1-3 are lists of things to fix. This is the story they add up to, written once so the manuscript
rewrite has something to aim at. Where this disagrees with parts 1-3, this wins.

Built 2026-07-28 from the argument map and a hostile-reviewer pass over all four documents.

---

## The paper in plain words — write this out before anything else

**The industrial partner's model stays exactly as it is. We asked whether its errors could be fixed by
tackling the class imbalance in their data. They cannot. The errors come from two things: mistakes
pile up at the edges between land types, and the model cannot reliably tell the two grassland types
apart.**

Say it that way first, then add the machinery. Two corrections to how this was being said:

- **"grade into each other" is jargon.** It means there is no clear line between them on the ground:
  one field becomes the other gradually, so there is no edge to draw. Write that.
- **The main confusion is the two grasslands with EACH OTHER**, not with cropland. Improved and
  semi-natural grassland account for about half of all error, running both ways. Cropland is a
  separate and much smaller problem - roughly 5% of total error - which only looks dramatic per class
  because cropland is small and about half of its pixels are called grassland. Do not merge the two
  into one sentence; they are different problems with different remedies.

## The framing for the literature

The standard aerial benchmarks delete the boundary band before scoring, naming label quality as the
reason and never justifying the width. We measure what that removes, on operational rural imagery.

## The arc, in order

**1. The field discards this measurement by convention.** ISPRS 2D Semantic Labeling erodes a 3-pixel
disc around every boundary and excludes it from scoring, naming label quality as the reason and
offering no measurement to justify the width. Two literature passes found no land-cover segmentation
paper reporting accuracy as a continuous function of distance to a class boundary. **Hedge this in the
paper** - "we are not aware of", not "nobody has". Volpi & Tuia DID compare eroded against
non-eroded ground truth and concluded boundaries are "often blurred within the 3 pixel erosion
radius", so the band has been measured at one width. What is absent is the ratio stated as such, and
the curve. The benchmark dates
from around 2012, so the convention has stood for over a decade unquantified. (Do NOT write "twenty
years" - that conflates it with the older proxy work, Smith 2002 and van Oort 2004, which uses
neighbourhood heterogeneity and patch size rather than boundary distance.)

**This is contribution 1, and it replaces the current one.** "We introduce a proprietary dataset" is
not a contribution — the data cannot be released, so a reader gets nothing from it. What a reader gets
is the measurement the benchmarks throw away.

**2. We measure it as a curve, not at a chosen width.** Error rate against distance to the nearest
ground-truth contour, swept across band widths. The shape is the evidence: the ratio rises steeply as
the band narrows, which is what boundary-localised error predicts and what a single width would hide.
No threshold is applied to it, because none can be justified.

**3. Two standard curation levers do not move it.** Cross-dataset pre-training and a class-balanced
sampler, crossed in a 2x2 over ten seeds on a fixed architecture. The design gives the transfer arm
twice the in-domain gradient steps, so its measured effect is an **upper bound** — and an upper bound
is exactly what the argument needs. Even crediting the whole procedure with everything, the gain is
modest. A confound that inflates cannot threaten a claim of smallness.

**4. The error is not spread across classes. It is one pair.** About half of all foreground error is
grassland confused with semi-natural grassland, and it runs **both ways at nearly the same rate**.

**5. The symmetry is what rules out class imbalance.** Imbalance predicts lopsided flow toward the
majority class. The observed flow runs slightly the other way — the majority class is a net donor of
pixels, not a net absorber. This is the paper's strongest single result and it is currently absent
from the manuscript entirely.

**6. That pair has no clear line between them on the ground.** One field becomes the other gradually -
management eases off toward wet corners, rushes come in - so there is no edge to draw and the annotator
had to put one somewhere. A forest edge is different: it is real, and the error there is mixed pixels.
**Avoid "gradational" and "grades into" in the paper.** Say there is no line on the ground.

**7. So for that pair, "the boundary is diffuse" and "the classes are hard to distinguish" are the
same fact, not two hypotheses.** This is interpretation and must be labelled as such — but it is what
makes the symmetry explicable rather than merely observed, and it is why the two accounts in step 9
are not fully separable even in principle.

**8. Therefore the recommendation is conditional.** Annotation effort helps where the edge is real and
the error is precision. It recovers nothing where the transition is gradational, because no placement
of the line is correct. The paper must not repeat the unconditioned "annotation effort is best spent
at boundaries" — that sentence is what led the industrial partner to conclude he should fund
annotation of the whole 8 m buffer.

**9. What we cannot say.** Whether that pair is inseparable in the imagery or was labelled
inconsistently. Separating them needs a second independent annotation pass over the same ground, which
does not exist. **State as a disjunction, never as a diagnosis** — and note that for a gradational
transition the two are not fully separable anyway.

**10. And the scope limit nobody has written yet.** One annotator, one protocol, one acquisition year.
What is characterised is the ceiling imposed by *that annotation process*, not a property of dense
land-cover annotation in general.

## Two supporting results that must be reported, not buried

**The near-infrared control.** The shipped model never sees NIR — the reader takes the first three
bands, forced by an ImageNet backbone. That invites the obvious objection that the useful band was
discarded. It was tested: a four-band variant moved semi-natural by under a point. Report it as a
**bounded null** with all three caveats (withdrawn split, three seeds, cannot use OpenEarthMap at all
so its control arm is sampler-only). It converts an objection into a stated negative control.

**Terrain is a geographic shortcut.** Elevation appears to separate the pair strongly, but the effect
reverses within a single tile containing both classes — it identifies which site a tile came from, not
which class a pixel is. This survives intact and is a clean piece of work.

## The title has to change

`CLAUDE.md` still calls the paper "*Diagnosing* a Label-Quality Ceiling". The paper explicitly declines
to diagnose, at step 9. A reviewer who reads the title and then the limitations has found the central
overclaim in ninety seconds. Locating, characterising or quantifying — not diagnosing.

## What this arc requires, and what pays for it

**Cut** — roughly four to five pages and five figures, all supporting no conclusion in the arc above:
the residual-uncertainty section and its three figures (the paper itself concedes the label-only
stratifications are stronger evidence); the confident-learning appendix (its headline is a retracted
statistic and its stated assumption is violated by the spatial structure the paper claims); the
frequency-versus-difficulty figure (the claim is refuted — rarity *does* predict difficulty here,
Spearman +0.70); the schematic mitigation-axes figure; the TTA row; the roads/topology aside.

**Add** — steps 4, 5, 6, 7 and the NIR control, none of which are in the manuscript.

**Fix** — every number in Results is from the withdrawn split. Methods describes the new design;
Results describes the retracted experiment. That is the largest single job and it is not optional.

## The four sentences to strike first

All four are in the abstract, highlights and contributions, all four are on the forbidden list, and
striking them takes half an hour:

- "label quality, rather than class imbalance or model capacity, is the dominant remaining constraint"
  (contribution 3, and again in the abstract)
- "indicating a label-quality ceiling rather than a limit of model capacity" (highlight)
- "annotation effort is best spent there" (highlight, unconditioned)
- "the consistency of that supervision, not the capacity of the model, becomes the binding constraint"
  (`main.tex:459`)

**Note the trap, now fixed in parts 1 and 3:** those documents pointed at `main.tex:459` as the
sentence to *preserve*. It is the forbidden one. The compliant sentence — *"such models blur edges
even on clean labels, so no single measurement separates the two. We read this as convergent evidence
rather than proof"* — is at **`main.tex:471`**, and it is the most protective sentence in the paper.

## Two measurements still worth running, both CPU-only on existing dumps

**The registered second arm.** The preregistration names a falsifiable necessary condition: the
near-boundary rate should stay roughly flat across the four cells while the interior rate falls, in
relative terms, with a stated falsifier. It has never been computed — four per-cell files exist and
nothing compares them. **If the paper cites its own preregistration and then does not report the
registered analysis, that is worse than a negative result.** Compute it, or say plainly that it was
not completed.

**Rho per seed rather than from the ensemble.** The current statistic is computed from the ten-seed
ensemble argmax. An ensemble is strictly better than its members and removes interior error
preferentially, so it inflates the exact ratio being claimed — the identical mechanism that
disqualified the `lift` statistic. Report per-seed mean and spread.

## The takeaway, in the author's words, and what licenses it

**"The popular fixes do not patch this. What is left is the architecture, the labelling, or different
signals."** That is the honest summary and it should shape the Conclusions.

What licenses each of the three, and the limit on each:

- **The architecture** — untested by design. It was held fixed so that every contrast was a data
  contrast. So this is a direction, not a recommendation: we can say curation did not move the error
  on a fixed architecture, not that changing the architecture would.
- **The labelling** — licensed only where the edge is real. At sharp boundaries the error is precision
  and better tracing helps. At the gradational grassland transition no placement of the line is
  correct, so annotation effort recovers nothing. The recommendation must carry that split or it
  repeats the sentence that sent the industrial partner off to fund the whole 8 m buffer.
- **Different signals** — one was tested. Near-infrared is the most vegetation-diagnostic band
  available and adding it moved the pair by under a point. Everything else the literature relies on for
  this distinction (texture at sub-metre, multi-temporal phenology, SAR, red edge) is untested here and
  the sensor cannot supply the last two at all. Name them as untested rather than leaving them silent.

**Do not over-generalise the "no boundary on the ground" point.** It is true of improved versus
semi-natural grassland, where management intensity grades off over metres. It is NOT true of forest,
settlement or water edges, which are real and where the residual error is mixed pixels. The paper's
value is in distinguishing the two cases; collapsing them throws that away.
