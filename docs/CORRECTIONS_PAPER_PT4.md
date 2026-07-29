# Part 4 — the narrative we are going with

> **SUPERSEDED IN PART. Read `notes/NARRATIVE_LITERATURE_FINAL.md`** — not `docs/NARRATIVE_FINAL.md`,
> which this banner named until 2026-07-29 and which quotes rho on the 294-tile population. Where any
> two disagree, the literature narrative wins. **What survives is listed below, with one exception:
> the ceiling-dependent passages at `:103–105`, `:220` and `:247` are withdrawn with it.** Note in
> particular the NIR control at `:107–111` and `:134`, which no successor
> document carries. Three things here are now known to be wrong and are corrected there: the
> claim that the confusion runs against the majority class (it does not, once class sizes are
> accounted for), the "near-symmetric" wording, and the framing of the whole paper on a null result.
> The per-class factorial decomposition, the registered across-cell arm and the sampler's measured
> effect had not been computed when this file was written. What survives intact: the boundary
> literature framing, the NIR and terrain material, the conditional recommendation,
> and the data-curation framing at the end.

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

---

## Why the factorial includes OpenEarthMap, and why its null is evidence

**Settled 2026-07-28** after questioning whether the factorial should have been run without it.

**Keep it.** The factorial tests the two things a practitioner reaches for when a model underperforms
on rare classes: get more data, or rebalance the data you have. Dropping OpenEarthMap leaves a
one-factor study asking only whether a sampler helps, which is thinner and licenses nothing. Showing
that neither standard move shifts the error is what permits moving on to diagnosis; without it the
paper only describes where errors sit, with no evidence that the obvious fixes fail.

**The limitation, stated plainly because a referee will find it in the mapping table.** OpenEarthMap
supplies **no labels for cropland and none for semi-natural** - the two classes carrying nearly all the
error. Three OEM classes (Bareland, Rangeland, Agriculture) collapse onto Grassland. So the
intervention cannot help those two classes by label transfer at all, only through general
representation.

**The observed pattern is exactly what that predicts**, which is why it reads as a result rather than a
puzzle. Paired per seed on Test A, OEM pre-training gives small consistent gains on Settlement (9/10
seeds) and Forest (8/10) - the classes whose mappings are clean at 80-96% - and nothing resolvable on
cropland or semi-natural.

**And this is the part to lead with, because it turns the null into evidence.** OpenEarthMap is the
largest public land-cover dataset in this space and it has no class corresponding to semi-natural
grassland. Its nearest, rangeland, lands 53% on improved grassland and only 15% on semi-natural when
measured against our labels. So the distinction this model cannot make is one that **public land-cover
data does not encode either** - a globally trained model, built by other people on other imagery, also
fails to separate these two grasslands. That is independent support for the ceiling, from a different
direction, and it explains the null instead of apologising for it.

**The arc, therefore:** try more data - the public data does not contain the distinction. Try
rebalancing - showing the model more of a class it cannot identify does not help. Neither works, and
for the first we can say precisely why. So the question becomes where the error actually sits: at
boundaries, in one class pair, whose transition has no line on the ground.

**What this costs:** one sentence in §2 saying OpenEarthMap supplies no labels for the two weakest
classes, and one in the Discussion saying the mapping was grounded on where the teacher's predictions
fell rather than on what the classes mean. Both are honest, and both pre-empt the criticism rather than
waiting for it.

**Not done and not worth doing before submission:** remapping rangeland to semi-natural (it is 53%
grassland, so that swaps one mostly-wrong label for a more-wrong one) or soft-label pre-training
against the confusion distribution the hard mapping discards. The second is genuine future work - the
matrix that grounds the mapping already contains the uncertainty the mapping throws away.

---

## The framing for the Data Curation special issue

**The curation decisions that mattered were not the two we tested.** Ranked by how much each actually
determined the outcome:

1. **The class definitions.** Semi-natural against improved grassland is a distinction the imagery does
   not support and public land-cover data does not encode. That decision was made before any model
   existed and it set the ceiling.
2. **The taxonomy mapping.** Translating one dataset's classes into another's decided which classes
   transfer could help at all. Three OpenEarthMap classes collapse onto Grassland; the two weak classes
   receive nothing. That single choice explains the whole per-class pattern.
3. **The two interventions.** Tested properly over ten seeds, neither moves the error.

So the paper's contribution to data curation is not "we tried two curation methods". It is that **the
curation decision with the most leverage was the earliest one - the label schema - and it is the one
nobody revisits.** More data and rebalancing are downstream of a taxonomy that already decided what is
learnable.

**What this licenses telling a practitioner**, which is the point for an industrial audience:

- Finer imagery will not fix the grassland pair; the distinction is not a resolution problem.
- More public data will not fix it either; the class does not exist in the public taxonomy.
- Annotation effort helps where there is a real edge to trace, and recovers nothing where the
  transition has no line on the ground.
- The class definitions themselves may be asking for a distinction the sensor cannot deliver. That is
  the question worth revisiting before spending on any of the above.

**Consequence for length.** This framing needs room in the Discussion, and it is more valuable than
per-cell detail on the factorial. If pages are needed, compress the factorial and the OpenEarthMap
mechanics - report the contrasts and the per-class pattern, drop the interaction interpretation and
the pre-training stage detail. The methods must stay clear; the ablation arithmetic does not.

---

## Added 2026-07-29 — two possible strengthenings, neither attempted

Both came out of settling the boundary argument. Neither blocks the rewrite; both are cheap and both
answer a question a reviewer is likely to ask.

**1. Measure each error patch's width, not only its area.** `component_sizes` returns pixel counts times
pixel area and nothing else, so a long thin ribbon and a compact blob of equal area are
indistinguishable. That is why the tenth-of-a-hectare threshold turned out to separate nothing — error
smeared along the reference boundaries clears it too — and why the paper now quotes the hectare
threshold instead.

Width comes straight out of a distance transform run *inside* each patch: for every pixel in the patch,
the distance to the nearest pixel outside it. The largest such distance is the radius of the biggest
circle that fits, so twice it is the width at the patch's widest point. A two-metre ribbon scores 1 m
however long it runs; a one-hectare blob scores tens of metres. That separates the two cases outright
rather than by proxy, and it is one call to `scipy.ndimage.distance_transform_edt` on the patch mask.

**2. Mosaic the predictions before measuring patches.** Scoring runs tile by tile, so every patch is cut
at the tile edge — no patch can exceed about 6.5 ha, and over 95% of the reference grassland mass on the
scored chips sits in regions clipped by an edge. Every size the paper reports is therefore the size of a
*piece* of a field. Reassembling the predictions onto real coordinates before labelling components would
give true patch sizes and would let the paper say "field" honestly.

This is not currently done because the per-seed predictions live on the cluster rather than locally. It
is a logistics limitation, not a scientific choice, and the paper should say so plainly in Limitations
rather than leave it to be found. It is also the obvious first move if the paper gets a revise.
