# The methodological choices, in plain language

Every deliberate choice this study makes, written so a non-specialist can follow it, with what was
chosen, why, and what it costs. **This is the input to a design review — it is a description, not a
defence.** Where a choice is contested or uncertain, that is said.

Nothing has been trained on the current split. Everything below is fixed *before* results exist,
which is the only thing that makes a stated limitation defensible rather than an excuse.

---

## A. What kind of study this is

**A1. The model is fixed and is not the subject.**
FT-UNetFormer is the industrial partner's deployed model. We never change it. Every comparison is
between *data* choices at a fixed model. Cost: no architecture comparison, so we cannot say this
model is good or bad — only what data interventions do to it.

**A2. Re-labelling the data was ruled out.**
Expert inspection reportedly found errors in ~88% of inspected masks. Fixing ~2,300 tiles by hand was
judged infeasible. So we work with the labels as they are and ask what *label-free* levers achieve.
Cost: the ground truth we score against is the same imperfect ground truth we train on.

**A3. The contribution is a diagnosis, not a method.**
We are not proposing a new technique. We are arguing that the remaining error is set by **label
quality at class boundaries**, not by model capacity or class imbalance. Cost: the paper lives or
dies on the diagnostic evidence, not on a headline accuracy number.

## B. The experiment

**B1. A 2x2 factorial: two interventions, on or off, all four combinations, ten seeds each.**
The interventions are (A) transfer from OpenEarthMap, a public land-cover dataset, and (B) a
class-balanced sampler that shows rare classes more often. 40 runs. This lets us separate what each
does and whether they interact, rather than testing them one at a time.

**B2. The four cells are NOT a linear pipeline — two of them branch.**

| cell | what actually runs | Biodiversity epochs |
|---|---|---|
| baseline | one training run, from the ImageNet→ADE20K backbone | 45 |
| sampler-only | one training run, sampler on | 45 |
| transfer-only | pre-train on OpenEarthMap+Biodiversity **combined**, then fine-tune on Biodiversity | **90** |
| full | the same pre-train, then fine-tune with the sampler on | **90** |

The stage names (`stage1`, `stage2a`, `stage2b`, `stage3`) suggest everything flows through every
stage. It does not: only the two transfer cells run the pre-training. **This is misread easily and
needs a diagram in the paper.**

**B3. The transfer arm therefore sees the Biodiversity training data twice, the others once.**
The pre-training pool is 3,190 tiles, and 1,072 of them *are* the Biodiversity training tiles. So
"transfer" delivers cross-dataset learning **and** a second pass over our own data, together.

*Why not remove the second pass?* Pre-training on OpenEarthMap alone would mean pre-training sees
**zero Cropland and zero Semi-natural** — measured, 0.000% of both — so two of five classes would get
no signal at all. That trades a confound we can declare for a defect we cannot.

*Why not give the baseline a second pass too?* That would make the comparison clean. It costs ten
extra runs, roughly 200 GPU-hours, +25%. **This is the live question.** Current decision (D12) is to
declare rather than fix, on the grounds that the transfer magnitude is setup, not the contribution.

**B4. The transfer arm gets two checkpoint-selection passes; the others get one.**
Every stage keeps its best epoch by validation score. The transfer cells select once during
pre-training and again during fine-tuning. Declared, not corrected.

**B5. Ten seeds vary the random parts, not the data.**
Each seed redraws the decoder initialisation, the sampler order and the augmentation stream. The
pre-trained backbone is identical across all four cells, which is what makes the comparison fair.

## C. How the data is split

**C1. Held-out ground must be genuinely held out — the previous split was not.**
Tiles overlap by 50%, so a random tile split put ~93% of each "held-out" tile's ground into training.
That campaign was withdrawn entirely. The split is now cut geographically.

**C2. One inland site is cut along a single axis, with gaps.**
`train | 256 m gap | val | 768 m gap | test`. Tiles straddling a gap are dropped. The gaps exist so
training and test ground are not neighbours.

**C3. Two upland sites are held out whole, as a separate test set.**
Test A = new ground inside a surveyed area. Test B = terrain never surveyed. **They are never
pooled**, because they answer different questions.

**C4. A split is accepted on how many places a class appears, not on what percentage it is.**
A class at 7% of pixels concentrated in three spots is unusable; 1.9% spread over eleven is fine.
Note: this criterion is applied to train/val/test only. **Test B has no such floor**, and its
Cropland sits in just 4 grid cells.

**C5. Colombian and Danish tiles in the delivered data are excluded.**
Different biome, different field structure. The study is three Irish sites out of eleven in the pool.

## D. How uncertainty is reported

**D1. Uncertainty is over training runs, not over ground.**
We report how much a number moves across the ten seeds, paired within each seed. We do **not** put a
spatial confidence interval on anything. The reason: both test sets are complete enumerations — every
tile scored, every pixel counted — so there is no sample to resample. A block bootstrap that used to
do this was removed on 2026-07-26.

*Cost, stated plainly:* we cannot say "this is how the number would behave on new ground". We say
"this is how it moves when you retrain on this ground". Two purposively chosen upland sites never
supported the stronger claim.

**D2. Paired means compared within the same seed.**
Seed 47's luck affects all four of its cells equally, so subtracting within a seed cancels it. This is
now applied to **all five classes plus mIoU** — it previously covered only three, which reported the
sampler's benefit paired and its cost unpaired.

**D3. No thresholds anywhere.**
A pre-registered "rho must exceed 4.0" bar was withdrawn because nobody could justify 4.0. Class
support verdict labels were withdrawn for the same reason. Numbers are reported; the reader judges.

**D4. We will not write "the sampler is redundant".**
If the sampler moves the score by a few tenths of a point, the honest statement is practical, not
statistical: *too small to justify the added complexity in a deployed pipeline*, with the number
shown. Claiming "no effect" as a fact would need a bar for what counts as nothing — the same invented
bar D3 refuses.

## E. How the boundary claim is measured

**E1. A band around every ground-truth class boundary, 8 metres wide.**
We compare the error rate inside that band with the error rate outside it. 8 m is an a-priori choice,
stated as such. It cannot be cited to prior work: the nearest paper's "8" is 8 *pixels* on small
photographs, and the numeric coincidence is just that.

**E2. The band is drawn from the ground truth only, never from predictions.**
Known cost, and it must be stated: this measure is not symmetric — it is more forgiving of predictions
that are larger than the true object than of ones that are smaller.

**E3. Tiles with no boundary at all are excluded.**
19 of 191 upland tiles are single-class, so a "near-boundary" rate is undefined for them. Including
them would drag the number down for a reason unrelated to the claim.

**E4. The primary evidence is a curve, not a single number.**
Accuracy as a function of how much boundary you exclude. The single-number summary (rho) is reported
descriptively alongside it.

**E5. The strongest form of the argument is a comparison, not a level — and it is a necessary
condition, not a diagnosis.**
If error is limited by labels, the near-boundary error rate should stay roughly flat across all four
cells while the interior rate falls. All four cells are scored on identical pixels, so the landscape
cancels out of that comparison entirely.

**Two qualifiers that must travel with the claim.** The comparison is *in relative terms*: the
near-boundary rate must vary by less than the interior rate does. And the falsifier: if both fall
proportionally, the concentration is a property of model quality and the label-ceiling reading is
not supported.

**What it cannot do.** Observing the pattern does not establish the premise. Every rival cause that
is constant across the four cells predicts the same thing — encoder/decoder edge blur, mixed pixels
at 0.5 m, image-to-vector registration offset — and since the architecture is fixed in every
contrast, none can be excluded here. `main.tex:471` already says this correctly (NOT 459 -- 459 carries the FORBIDDEN "binding constraint" sentence and must be struck) and that sentence
should survive: *"such models blur edges even on clean labels, so no single measurement separates
the two. We read this as convergent evidence rather than proof."*

## F. Data handling

**F1. OpenEarthMap classes are mapped onto ours by measurement, not by judgement.**
A model trained on OpenEarthMap is run over our training data, and its confusion decides the mapping.
Consequence: Bareland, Rangeland and Agriculture all land on Grassland, so OpenEarthMap contributes
**no Cropland and no Semi-natural** at all.

**F2. Foreground mIoU excludes Background.**
Averaged over the five real classes. Background is 1.7% of training pixels but 38% of Test B.

**F3. Training stops at 45 epochs.**
The learning-rate schedule restarts at 15 and 45, so stopping anywhere else ends mid-cycle.

---

## The open questions this review should settle

1. **B3** — declare the extra Biodiversity pass (free), or buy the clean comparison (10 runs, ~200
   GPU-h)? Now is the only cheap moment.
2. **B2/B4** — is "main effect A" honestly describable as *transfer*, or must it be named
   *transfer-plus-a-second-pass* everywhere it appears?
3. **C4** — is it defensible that Test B faces no support floor, given Cropland sits in 4 cells?
4. **D1** — is "no spatial interval anywhere" right, or does a reviewer expect one enough that its
   absence needs more than a paragraph?
5. **E2** — does ground-truth-only banding bias the *comparison between cells*, or only the level?
6. **A2/A3** — the whole contribution rests on labels being the binding constraint. What evidence for
   that exists beyond the ~88% inspection figure?
