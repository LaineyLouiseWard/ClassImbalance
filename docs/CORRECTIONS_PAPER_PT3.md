# CORRECTIONS PART 3 — the narrative, rebuilt on the campaign results

Written 2026-07-28 from five of ten finished seeds, after two adversarial reviews refuted most of an
earlier draft narrative. Parts 1 and 2 stand; this file supersedes
`notes/rebuild_2026-07/for_the_paper/NARRATIVE_2026-07-28.md`, which contains claims listed as unsafe
below. Numbers here are five-seed means on Test A unless stated, and provisional until all ten land.

---

## 1. What is actually established

**One class pair carries about half the error.** Grassland predicted as semi-natural is 26.4% of all
foreground error; semi-natural predicted as grassland is 22.6%. Together, 49% of every mistake. No
other pair is close: forest and grassland together are 29%, everything else is small.

**The confusion is close to symmetric, and that is the load-bearing fact.** Class imbalance predicts
lopsided flow toward the majority class. The observed flow runs slightly the *other* way — grassland
is a net donor of pixels to semi-natural, not a net absorber. An imbalance account does not predict
that.

**Boundary error is concentrated, and it strengthens as the band narrows.** Roughly 4x the interior
rate within one to two pixels, about 2x at 8 m, consistent across all five seeds. This survives the
leakage fix and the preprocessing fix.

**Both interventions are small on Test A.** Cross-dataset pre-training gives a modest mIoU gain;
the class-balanced sampler gives nothing. Neither result is in doubt; their per-class attributions are
(see §2).

## 2. What was claimed and must NOT be written

Refuted on the data by adversarial review, 2026-07-28:

| claim | why it fails |
|---|---|
| "Grassland absorbs the minority vegetation classes" | Backwards for two of three. Grassland is a net donor to semi-natural and forest. |
| "Cropland into grassland is the largest confusion" | It is fifth, ~5% of total error. It is 52% *of cropland*, but cropland is 1.35% of the test set. |
| "Rarity does not predict difficulty" | Spearman between train share and IoU is +0.70, rising to +0.87 on block support. Rarer *is* worse. |
| "The rarest class is among the strongest" | True on pixel share only. Settlement appears in 724 of 1,072 training tiles against cropland's 248 — it is pixel-rare but scene-ubiquitous. |
| "The sampler increased absorption" | Per-seed differences swing +/-16 to +/-29 pp around a mean of +3. The sign is not stable. |
| "Transfer reduced absorption without supplying labels" | Mixed on Test A; decisively worse on Test B. And the arm trains on the in-domain data twice, so it does add exposure. |

**The general lesson:** row-normalised confusion percentages make a small class look dominant. Always
check absolute pixel volumes and net flow before asserting a direction.

## 3. The separability evidence needs fixing before it can carry weight

**NDWI here is an open-water index.** The script computes (Green - NIR)/(Green + NIR), which is
McFeeters (1996), designed to delineate lakes. The canopy-water index is Gao (1996) and requires SWIR,
which Pleiades does not carry. Reporting that it "separates the pair even less" is not a null result;
it was never a moisture measurement. **Reframe as a missing-band limitation, cite both papers.**

**Texture was never tested, and it is what a referee will ask for first.** At 0.5 m, structure is the
expected discriminator: semi-natural swards are tussocky and heterogeneous, improved swards smooth and
often mown in patterns. The precedent is already in the library — Dimitrov et al. (2024) use GLCM
texture on **Pleiades** imagery and gain 1-8% accuracy across four single-date classifications.

**Cohen's d is not this field's separability statistic.** Jeffries-Matusita distance and transformed
divergence are, and both are multivariate. Cohen's d is univariate and mean-only, so two classes with
equal means and different covariance score zero and are still separable.

**Consequence for the wording.** "Not separable from a single image" is not supported. What is
supported: *per-pixel spectral indices from this sensor do not separate the pair.* The sensor has no
red edge and no SWIR, and the multi-temporal and radar cues the literature relies on are unavailable
here, so the negative result is partly structural.

**Also fix:** the script's docstring still describes the withdrawn split (219 tiles,
`biodiversity_split`) while the code reads `split_f1/val` (173 tiles), and no current-split output
exists — the archived JSON is from the withdrawn campaign.

## 4. The honest narrative, and its limit

The residual error is dominated by one class pair that the model cannot separate, that simple spectral
indices cannot separate, and that fails symmetrically rather than collapsing into the majority class.

**Two accounts remain open and this study cannot choose between them:**

1. The imagery genuinely does not carry the distinction at this sensor and this date.
2. The boundary between the two classes was drawn inconsistently during annotation.

**Distinguishing them needs a second annotation pass over the same ground, which does not exist.**
`DO_NOT_ADD.md` already forbids claiming a measured inter-annotator ceiling, and that prohibition now
matters more than before: the symmetric confusion is the strongest evidence in the paper and it is
exactly the evidence that cannot be attributed.

**So the claim must be stated as a disjunction, not a diagnosis.** The paper's contribution is
locating the ceiling and showing that two standard data-curation levers do not move it — not proving
which side of the imagery/annotation line the ceiling sits on.

## 5. Before submission, in priority order

1. **Reframe NDWI** as a missing-SWIR limitation. Writing only, unavoidable, prevents an easy catch.
2. **Narrow the separability claim** to per-pixel spectral indices from this sensor.
3. **Fix the stale docstring** and regenerate the separability output on the current split.
4. **Classifier probe on the four raw bands**, semi-natural vs grassland, tile-blocked cross-validation,
   reporting AUC. Answers the actual question rather than testing two chosen ratios. About an hour.
5. **GLCM texture**, if time allows. This is the analysis that closes the referee's first objection,
   and the Pleiades precedent is already in the library.
6. **Test the cropland/grassland pair too.** The probe has only ever covered one pair.

---

## 6. The near-infrared band: dropped by the pipeline, and tested — it does not rescue the pair

**The shipped model never sees near-infrared.** `geoseg/datasets/biodiversity_dataset.py:130` takes
`data[:, :, :3]`, keeping the three visible bands and discarding band 3 (NIR). This is forced by the
backbone: the Swin encoder is ImageNet-pretrained with `in_chans=3`.

**This creates an obvious objection** — the separability probe computes NDVI, which is built FROM
near-infrared, so it measures information the model is never given. "The imagery cannot separate these
classes" and "the model cannot separate these classes" are therefore different statements, and the
model is working with less than the imagery offers.

**The objection was already tested, and the answer is that NIR does not rescue the pair.** A four-band
variant was implemented and run (`notes/RGB_NIR_EXPERIMENT_PLAN.md`, branch `experiment/rgb-nir`),
inflating the first conv from 3 to 4 channels and seeding NIR from the red channel. Three seeds, on the
old split:

| variant | mIoU | semi-natural IoU |
|---|---|---|
| RGB+NIR, clsbal | 0.8718 | 0.902 |
| RGB, sampler-only (the plan's stated control) | 0.8811 | 0.894 |
| RGB, clsbal | 0.9007 | 0.912 |

Semi-natural moves by under a point in either direction and mIoU falls slightly. **Adding the band most
diagnostic for vegetation did not resolve the confusion.**

**Three caveats that must travel with it.** The numbers are from the withdrawn leaking split, so only
the contrast is usable and not the levels. Three seeds, not ten. And the four-band variant cannot use
OpenEarthMap transfer at all, because OpenEarthMap is RGB-only — it initialises from the ADE20K stem
instead, which is why its own plan nominated sampler-only as the control rather than clsbal.

**Why this is worth reporting rather than burying.** It converts a reviewer's obvious objection ("you
threw away the useful band") into a stated negative control. The claim becomes: the pair is not
separated by per-pixel spectral indices, and is not separated by a model given the near-infrared band
either. That is much stronger than the spectral probe alone, and it is the honest reason the
band-dropping is a limitation rather than an explanation.

---

## 7. The one measurement that turns the description into a diagnosis

**Decided 2026-07-28.** §3 and §5 above proposed a ladder of probes, indices and texture. That is a
different paper. What this one needs is a single sentence: *the pairs the model confuses are the pairs
the imagery cannot distinguish.*

**One statistic, three pairs, four raw bands, no indices.**

| pair | separability | share of model foreground error |
|---|---|---|
| grassland – semi-natural | to measure | 49% |
| cropland – grassland | to measure | ~5% |
| cropland – semi-natural | to measure | small |

**Why the raw bands and not indices.** Every vegetation index is a deterministic function of the four
bands, so the bands set the ceiling on per-pixel spectral separability and no index list can exceed
it. Using them directly also removes seven index definitions from a manuscript that is already long.

**The statistic is Jeffries-Matusita distance**, the field's standard answer to "are these classes
separable", computed from each class's mean vector and covariance over the four bands:

    B  = 1/8 (m1-m2)' [(S1+S2)/2]^-1 (m1-m2)  +  1/2 ln( |(S1+S2)/2| / sqrt(|S1||S2|) )
    JM = 2 (1 - exp(-B))

Range 0-2; conventionally >1.8 is good separation and <1.0 poor. **State the convention** — some
sources define JM with a square root and range 0-sqrt(2). No model is fitted and nothing is trained.

**What each outcome means.** If the separability ranking matches the confusion ranking, the model's
errors track what the imagery carries, and the ceiling is the data. If a pair the imagery separates
well is nonetheless confused, the ceiling is the model or the labels for that pair. Both outcomes are
reportable; only the comparison makes it diagnostic rather than descriptive.

**The limit, which must be written with the claim.** A separability statistic at chance is *consistent
with* the information being absent but never proves it — no finite measurement does. The defensible
wording is "no evidence of exploitable separability in the per-pixel spectra", never "the information
is not there".

**Explicitly out of scope** and to be stated as untested if a referee asks: GLCM texture, additional
spectral indices, object and shape features, multi-temporal and SAR cues. The last two are unavailable
on single-date Pleiades in any case.

**Superseded by this section:** the probe ladder in §5 items 4-6. §3's defects (the open-water NDWI,
the stale docstring, the unverified band order) still need fixing regardless.

---

## 8. DECISION 2026-07-28: no separability statistic in this paper

§7's plan is withdrawn and `scripts/analysis/class_pair_separability.py` is deleted. The statistic
itself was sound and self-tested, but a review of its assumptions found two design traps serious enough
to manufacture the very result it was meant to test:

- **Prior confound.** Jeffries-Matusita is prior-free; a confusion matrix is not. Correlating the two
  would have largely measured class frequency rather than separability. The comparison would need a
  symmetrised, row-normalised confusion measure, not the raw error share.
- **Boundary common cause.** Mixed pixels at 0.5 m deflate separability specifically for spatially
  ADJACENT class pairs, and adjacency is also what drives the model's confusion. The correlation could
  therefore appear for a reason unrelated to the claim — in a paper whose headline finding is
  boundary-concentrated error, that is the first thing a referee would test.

Both are fixable, but not defensibly on the day before submission by an author who has not used the
statistic before. Reported as untested rather than done badly.

**For a future revision, if wanted:** erode class masks before sampling, compute the confusion over the
same eroded interiors, rank on the Bhattacharyya distance rather than JM (which saturates at 2), assess
rank stability by leave-tile-out rather than by any interval over pixels, and state the 0-2 convention
explicitly. Note also that the ranking comparison appears genuinely uncommon in the literature — a
Scopus and OpenAlex search found no paper comparing a separability ranking against a model's realised
confusion — so the contribution would be the quantification, not the idea.

## 9. Why the error sits at boundaries: two mechanisms, and only one is fixable

**Interpretation, not measurement. Label it as such in the paper.**

The two boundary types in this landscape are not alike:

- **Sharp edges** — forest to grassland, settlement to anything. The edge physically exists; error
  there is mixed pixels and edge blur, and it is a precision problem that finer imagery or more careful
  tracing could reduce.
- **Gradational transitions** — improved grassland to semi-natural. There is no line on the ground.
  Management intensity falls off gradually toward wet corners, rushes encroach, and the annotator must
  impose a boundary that does not physically exist.

**For the grassland pair, "the boundary is diffuse" and "the two classes are hard to distinguish" are
the same fact.** They grade into one another, which is why there is no crisp edge to draw and why the
model cannot find one. These are not competing explanations to be tested against each other.

This also accounts for the near-symmetry of the confusion. A precision problem at a real edge produces
symmetric error; a definitional problem produces it along a broad transition zone, which is what the
distance decay shows.

**Consequence for the practical recommendation.** Annotation effort helps at sharp edges. It cannot
resolve a transition that has no true location, and spending there recovers nothing. The paper should
distinguish the two rather than recommending boundary annotation in general.

## 10. What stays open, and where it goes

Whether the grassland pair is inseparable in the imagery or was labelled inconsistently cannot be
settled here: it needs a second independent annotation pass over the same ground, which does not exist.
**Limitations and future work**, stated as a disjunction, never as a diagnosis.
