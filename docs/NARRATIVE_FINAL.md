# The final narrative

Written 2026-07-28 after the ten-seed campaign finished, the per-class decomposition was computed,
the registered second arm was run for the first time, and three review agents attacked the previous
version. **This supersedes `CORRECTIONS_PAPER_PT4.md` wherever the two disagree.** Numbers live in
`RESULTS_TEN_SEED.md`; nothing here introduces a number that is not in that file.

The old narrative was built on a null result. It said: we tried the two standard fixes, neither
worked, so the problem must be the labels. Three separate reviews broke that, and they were right to.
Ruling out one cause does not select another one, and the aggregate numbers were too noisy to rule
out much of anything. This version does not rest on a null.

---

## 1. The paper in plain words

A company maps Irish farmland from half-metre aerial photographs. Their model is good at forest,
grassland and built-up land, and poor at two classes: cropland, and semi-natural grassland (rough
grazing, rushy ground, the unimproved fields). Those two classes are also the rarest in their
training data, so the obvious diagnosis is class imbalance and the obvious fixes are to add more data
or to show the model more of the rare classes.

We tested both fixes properly and found something more useful than "they don't work".

**Adding a large public dataset does a lot. It just does not do what people assume, and the headline
number hides it completely.** Pre-training on OpenEarthMap changes each of our five classes by a
different amount and in different directions, and the sizes and the signs are set by one decision
made long before any model was trained: how the public dataset's classes were matched to ours. The
classes with a clean, dedicated match improved a great deal and in nearly every run. The class that
three different public classes were all folded into got dramatically worse, in every single run. The
two classes with no match at all did nothing. Averaged into one score, a large gain and a large loss
cancel, and the summary reads "no effect".

**Showing the model more of a rare class does very little, and we can show it is not because the
method failed to act.** Rebalancing made the model claim 24% more semi-natural ground than actually
exists. It moved the decision exactly as designed. Of every hundred extra pixels it claimed, three
were right.

**And the error that is left does not sit where either fix could reach.** Just under half of all
mistakes are the two grassland types being confused with each other, in both directions. The error
concentrates tightly at the lines between classes, and it stays there no matter which of the four
configurations we train. Across the four, the error away from boundaries moves around four times as
much as the error at boundaries does.

So the finding is about curation, and it is a positive finding, not an absence:

> **The curation decision that determined the outcome was the class definitions and how they were
> matched to the public data. Both decisions were made before any model existed, and neither of the
> two levers a practitioner reaches for afterwards can reach back and undo them.**

## 2. Why this replaces the old story

The old story was "two things failed, so it must be the labels". Four problems with it, all raised
independently by reviewers, all fair:

- **You cannot name a cause by eliminating one other cause.** Architecture, training budget, input
  bands, loss and resolution were never excluded, and the architecture was held fixed *by design*, so
  a limit caused by the architecture could not show up in any contrast we compute.
- **The nulls were not tight enough to be informative.** The confidence interval on either main
  effect is about plus or minus three points, and the whole spread between our four configurations is
  2.45 points. The intervals are wider than the thing being discussed.
- **One of the two "tests" could not have worked.** OpenEarthMap supplies no labels for the two weak
  classes. Claiming the null as evidence while explaining in the next paragraph why the test was
  incapable is not an argument.
- **The strongest stated result pointed the other way once class sizes were accounted for.** See §4.

The new story fixes all four, because it rests on effects that are large, consistent across seeds,
consistent across two independent landscapes, and explained by a mechanism we can point at in a table.

## 3. The three findings, in order

### Finding 1 — the class mapping decides what pre-training does, and the average hides it

Pre-training on OpenEarthMap, measured on the upland test set, paired within each seed:

| our class | what the public data contributes to it | effect |
|---|---|---|
| Settlement | three source classes, matched at 52-88% | **+11.3 points, in 10 of 10 runs** |
| Forest | one source class, matched at 80% | **+8.5 points, in 9 of 10 runs** |
| Grassland | three very different source classes all folded in | **-19.4 points, in 0 of 10 runs** |
| Cropland | nothing | -2.7, in 5 of 10 — no effect |
| Semi-natural | nothing | -2.3, in 3 of 10 — no effect |

The inland test set gives the same ordering of signs at smaller sizes, so this is not one landscape's
quirk.

Overall score: **-0.9 points. "No effect."**

That is the paper's central object lesson, and it needs no proprietary data to be useful to a reader:
the mapping table is publishable, OpenEarthMap is public, and any practitioner planning to pre-train
on a public land-cover dataset has a mapping table of their own sitting in front of them.

Two things this also settles:

- **It answers the obvious methodological objection.** Our pre-training arm gets exactly twice as
  many training steps on the company's own data, so a reader can say any gain is just extra training.
  Extra training makes every class better. This made one class worse in every single run.
- **It ends the "neither intervention did anything" framing.** Pre-training did a great deal. It is
  the summary metric that did nothing.

### Finding 2 — rebalancing acts, and buys almost nothing

The class-balanced sampler shows the model semi-natural ground about 2.8 times more often than its
share of the data. The effect on the model's behaviour is exactly what it should be: the model goes
from claiming 16.2 million semi-natural pixels to 18.0 million, against 14.6 million that are really
there. It over-claims the rare class by 24%.

What that bought:

- semi-natural recall 56.03% to **56.40%**
- semi-natural precision 50.3% to **45.6%**
- semi-natural pixels wrongly called grassland: 39.15% to **39.80%**
- of every 100 extra pixels claimed, **3 were right**

**The model is not reluctant to say "semi-natural". It cannot tell which pixels are semi-natural.**
That distinction is the whole finding, and it is only visible because the intervention demonstrably
did its job.

The mechanism is countable. Semi-natural appears in 261 of 1,072 training tiles. Showing those 261
tiles more often adds repetitions, not new places. Across our five classes, per-class accuracy is
ordered almost perfectly by *how many distinct places a class appears in* (Spearman +0.98 against
950 m grid cells, +0.90 against tiles) and much more loosely by *how many pixels it has* (+0.60). With
five classes that is an ordering and not a fit, and it must be reported as one. But it says plainly
what neither lever can supply: **more places.** The sampler re-weights the places that exist, and the
public dataset contributes no places at all for these two classes.

### Finding 3 — the error that is left sits at the lines between classes, and stays there

Just under half of all mistakes — **46.68%** — are improved grassland and semi-natural grassland
being confused with each other, in both directions.

Error rate close to a boundary against error rate away from it, ten seeds, baseline:

| how close | error rate near | error rate away | ratio |
|---|---|---|---|
| within 1 m | 41.2% | 10.8% | **3.85** |
| within 8 m | 19.6% | 8.8% | **2.28** |

The ratio climbs as the band narrows. That shape is the evidence, and reporting one width would hide
it.

**The registered check, run for the first time on 2026-07-28, and it passes.** We committed in advance
to a necessary condition: across the four training configurations, the error rate near boundaries must
move less than the error rate away from them, in relative terms. It does, in every seed, at both
widths. At 8 m the near-boundary rate varies by 3.5% across the four configurations and the interior
rate by 14.1%, a gap of 10.6 points with a confidence interval of [8.3, 12.9] and **10 of 10 seeds
agreeing**.

Two honest qualifications that travel with it. The interior rate does not *fall* across the four — the
full model has the highest interior rate of the four. What is claimed is that boundary error is
immovable while interior error is not, which is the registered wording. And this is a necessary
condition, not a diagnosis: anything that is constant across the four configurations, including the
architecture's own tendency to blur edges, predicts the same flat boundary rate.

## 4. Four things that must not be written

**"The confusion runs against the majority class."** This was called the paper's strongest result in
the previous draft and it is wrong. It compared raw pixel counts between two classes that differ in
size by a factor of ten. Per pixel, a semi-natural pixel is called grassland 39.2% of the time and a
grassland pixel is called semi-natural 4.8% of the time. That is 8:1 toward the majority, which is
exactly what class imbalance predicts. Delete the claim.

What survives, and is enough:

- The classes predicted over their true extent are the two *rarest* in training. The class badly
  under-predicted, cropland, has nearly twice the training pixels of either. So under-prediction is
  not a function of rarity here.
- The confusion is imbalance-shaped, and rebalancing does not shift it (39.15% to 39.80%). **Error
  that looks like an imbalance problem, and that an imbalance remedy does not fix, is the finding** —
  it does not need to be re-described as something other than imbalance-shaped.

**"Near-symmetric."** The ratio is 1.26 on raw counts and 8:1 on rates. Neither is symmetry.

**"Rarity does not predict difficulty" — and equally, "rarity does predict difficulty".** With five
classes there is not enough to support a claim in either direction, and this project has now asserted
both. State the ordering and fit no line. This is a change from the earlier plan, which cut the
frequency-versus-difficulty figure on the grounds that its claim was refuted. The better reason is
that five points license nothing either way.

**Anything crediting pre-training with an overall gain.** On the inland test set it is -0.37 points,
positive in 5 of 10. Every sentence in the manuscript giving it +2.10 or +1.88 comes from the
withdrawn split and must go.

## 5. The one result that needs quarantining

The only contrast that clears significance on the inland test set is the interaction between the two
interventions: -2.08 points, negative in 9 of 10 seeds, p = 0.023. It is entirely a cropland effect.
Cropland alone accounts for -2.09 of the -2.08. Every other class is flat.

Cropland is 1.35% of that test set, in 52 of 294 tiles and 8 grid cells, which is the split's minimum
acceptance floor, with a between-seed spread of 0.195 IoU against a mean of 0.340. The sign also
reverses on the upland test set.

**So it is reported as a cropland effect with its support stated, or not at all.** Reporting it as a
general property of the two interventions while the per-class table says no cropland claim is
supportable is a contradiction a referee finds in one reading.

## 6. How this traces through the paper

The trace below assumes the section structure of the current `manuscript/main.tex`. Line numbers
drift; re-grep after the first edit.

**Title.** Drop "Diagnosing" — the paper declines to diagnose. Recommended:

> **The Taxonomy Mapping Decides the Outcome: Per-Class Effects of Cross-Dataset Pre-Training and
> Class Rebalancing in Rural Land-Cover Segmentation**

Alternative if that reads too strong: *"What Cross-Dataset Pre-Training Actually Changes: ..."*.

**Abstract.** Currently frozen by the author. Lift the freeze last, and rewrite around Finding 1. It
must contain the +11.3 / -19.4 / -0.9 triple, because that is the sentence a reader remembers. Strike
the three forbidden sentences it carries.

**Introduction §1.4 — the contributions.** All three are replaced.

1. Pre-training's effect on a fixed architecture is per-class, and its sign and size are predicted by
   the taxonomy mapping. Aggregate metrics hide this.
2. Class-balanced sampling is shown to act as designed and to buy almost nothing, with the arithmetic
   of what it bought.
3. The residual error is located: one class pair, concentrated at boundaries, and unmoved across all
   four configurations by a pre-registered test.

The current contribution 1 ("we introduce a proprietary dataset") goes. The paper concedes later that
the numbers cannot be replicated externally, so the dataset gives a reader nothing.

**§2 Methods — protect entirely.** The spatially blocked split, the buffer distances, the 950 m
support criterion, the deduplicated scoring subset, and the openly declared withdrawn campaign are
better than the field norm and are the reason a referee will believe any of the results. Do not
compress a word. The one addition owed here is a plain sentence saying that the mapping was grounded
on where the public model's predictions landed rather than on what the classes mean.

**§3 Results — rebuilt, in this order.**

1. Per-class factorial contrasts on both test sets, with the mapping table beside them. This is the
   new §3.2 and it is the paper's centre of gravity.
2. Aggregate contrasts with intervals, stated as bounds, with the sentence saying the intervals are
   wider than the spread between cells.
3. The sampler's mechanical effect: predicted area, recall, precision, and the three-in-a-hundred
   arithmetic.
4. The class pair, on absolute volumes *and* rates, with both stated.
5. The boundary curve per seed per cell at both widths, and the registered across-cell test.
6. The interaction, quarantined as in §5 above.

**§4 Discussion.** Two subsections carry the weight. First, why the mapping produced that pattern, and
what a practitioner should do about it — look at the mapping before pre-training, and read per-class
effects rather than the aggregate. Second, the conditional recommendation: annotation effort helps
where there is a real edge to trace and recovers nothing where there is no line on the ground, because
no placement is correct. The unconditioned version of that sentence is what sent the industrial
partner off to fund annotation of an entire 8 m buffer.

Keep the terrain result, which is clean: elevation appears to separate the two grasslands but the
separation reverses inside a single tile containing both, so it identifies where a tile is, not what a
pixel is.

**§4 Limitations.** One annotator, one protocol, one acquisition year, so what is characterised is the
ceiling imposed by *that* annotation process. Two purposive upland sites, so no statistical claim of
generalisation. No second annotation pass, so whether the pair is inseparable in the imagery or was
labelled inconsistently stays open, and is stated as an open question and never as a diagnosis. The
pre-training arm's doubled step count, declared rather than corrected, with the per-class pattern given
as the reason it cannot explain the result.

**§5 Conclusions.** The curation decision with the most leverage was the earliest one. More data and
rebalancing are both downstream of a taxonomy that already fixed what was learnable.

## 7. What to cut, and what that buys

Ranked by how much the paper improves, not by pages.

1. **The confident-learning appendix.** Its headline is a retracted statistic, its stated assumption
   is violated by the spatial structure this paper claims, and under the new framing it is the closest
   thing in the paper to a measured label-error bound. A referee will read it as the inter-annotator
   ceiling this project has never measured, which is the most damaging misreading available. About two
   pages, one figure, one table.
2. **The residual-uncertainty section and its three figures.** The paper already concedes the
   label-based analyses are stronger. Calibration is a different paper and it dilutes this one. About
   two pages, three figures.
3. **The frequency-versus-difficulty figure**, for the reason in §4 above.
4. **The mitigation-axes schematic.** It draws a data-versus-model split the design cannot test, since
   the architecture is fixed, and it promises the forbidden claim before the text arrives to hedge it.
5. **Test-time augmentation, and the roads aside.** Orphaned; nothing depends on them.
6. **Compress** the pre-training stage mechanics and the factorial arithmetic. Methods stay clear;
   bookkeeping does not need the space.

Roughly five pages and five figures, which is what the per-class results and the discussion need.

**Repurpose rather than cut:** the qualitative panel figure. Four configurations whose aggregate
differences sit inside the noise makes a figure showing nothing. But no reader can obtain this imagery,
and one panel is the only way they will ever see what improved and semi-natural grassland look like
side by side, and why there is no line to draw. Re-point it at the class pair.

## 8. The four sentences to strike first

All four are in the abstract, highlights and contributions, all four are on the forbidden list, and it
takes half an hour.

- "label quality, rather than class imbalance or model capacity, is the dominant remaining constraint"
- "indicating a label-quality ceiling rather than a limit of model capacity"
- "annotation effort is best spent there" — unconditioned
- "the consistency of that supervision, not the capacity of the model, becomes the binding constraint"
  (`main.tex:459`)

**Do not touch `main.tex:471`** — the convergent-evidence-rather-than-proof sentence. It is the most
protective sentence in the paper.

## 9. What a practitioner can take away

This is the part an industrial reader is buying, and it is now licensed by measured effects rather
than by a null.

- **Read your class mapping before you pre-train on public data.** It tells you which of your classes
  will improve and which will get worse. Folding several source classes into one of yours can cost you
  more than the others gain.
- **Do not judge pre-training on an aggregate score.** Ours said "no effect" while hiding an 11-point
  gain and a 19-point loss.
- **Rebalancing changes what the model claims, not what it can tell apart.** If the model already
  over-claims the rare class, more of it will not help.
- **What predicts per-class accuracy here is the number of distinct places a class appears in, not the
  number of pixels.** Buying more pixels of the same fields is not the same as buying more fields.
- **Annotation effort pays where there is a real edge to trace, and recovers nothing where the
  transition has no line on the ground.**
- **Some class definitions ask for a distinction the sensor cannot deliver.** That is the question
  worth revisiting before spending on any of the above.

## 10. What stays open, and is said so

Whether the two grasslands are genuinely inseparable in this imagery, or were labelled inconsistently,
cannot be settled here. It needs a second independent annotation pass over the same ground, which does
not exist. Stated as an open question in Limitations, never as a diagnosis, and noted that for a
transition with no line on the ground the two are not fully separable even in principle.
