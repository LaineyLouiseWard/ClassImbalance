# CORRECTIONS — verified, and they need to happen

> ## STATUS 2026-07-27 — worked in the manuscript pre-write chat
>
> Each correction below now carries a status line. Anything still open was moved to
> **`docs/CORRECTIONS_PAPER_PT2.md`** (written by the other chat — do not duplicate it here).
>
> | | |
> |---|---|
> | DONE | 1, 2 (subtractive half), 4, 6, 8, 9, 10 (a)(b)(c)(d)(e)(f)(h)(k)(l) |
> | OPEN → PT2 | 2 (replacement sentence), 3, 5, 7, 10 (g)(i)(n-code) |
> | STANDING RULE, nothing to do | 10 (j)(m) |
>
> Pages held at 28. `main.tex` compiles with all references resolved (eight figures are
> deliberately absent, so `latexmk` fails on a missing graphic — that is expected).

**Every item below was checked against the actual file, not inferred.** The line numbers, the
quotations and the supporting measurements were all opened and read on 2026-07-27. These are not
suggestions to consider; they are errors that are in the manuscript now.

## Scope of this file, and where the OTHER kind of item lives

This file is for **things that are wrong and must be deleted or changed**.

It is NOT the list of design properties the paper must *state*. Those live in
**`docs/METHODS_STATED_LIMITATIONS.md`**, and the distinction matters: a stated limitation is a
strength, an unstated one is what sinks a paper. Nothing below duplicates that file.

The properties that must be STATED, not corrected — go to METHODS for the measurement behind each:

| property | where |
|---|---|
| Test B has no per-class support floor; its Cropland occupies 4 grid cells | METHODS §4 |
| The transfer arm selects a checkpoint twice; the other cells once | METHODS §2 |
| The train/val gap is 256 m, below the inland 750 m composition range — validation is not spatially independent of training, and every checkpoint is selected on it | METHODS §2, §4 |
| The transfer arm receives exactly 2.00x the Biodiversity gradient steps | METHODS §1 |
| OpenEarthMap contributes zero Cropland and zero Semi-natural | METHODS §3 |
| The grid counts cells touched, not independent parcels | METHODS §6 |
| No spatial interval is reported; uncertainty is per-seed and paired | METHODS §7 |
| The study is three Irish sites of eleven in the delivered pool | METHODS §8 |
| Overlapping tiles are scored without deduplication, so ground is weighted 1–4x | METHODS §6 |
| Re-cutting the tiles without overlap was available and declined | METHODS §2 |

**The split, in one paragraph, because it is described inconsistently elsewhere.** The inland site is
an irregular survey polygon, NOT a rectangle — 1,952 tiles occupy 47 x 52 grid positions, so 20% of
its bounding box is empty and columns hold between 35 and 49 tiles. It is cut once along easting into
three strips: train | 256 m gap | val | 768 m gap | test. Tiles straddling either gap are dropped:
**413 tiles, 21.2% of the site.** Two upland sites ~58 km away are held out whole as Test B. That
gives **one training set, one validation set, and TWO test sets** — not three test sets. Validation is
not held out in the same sense: every checkpoint in every cell is selected on it.

## Read this before the corrections: the paper's argument is not damaged, it is strengthened

The 2x2's job here is to establish that the two interventions give **only modest gains**, so that
something else must be binding — and that something is the boundary/label ceiling. Everything after
the factorial depends on the gains being modest.

**The confound below INFLATES the reported transfer effect.** Part of what is being credited to
cross-dataset transfer is a second pass over the Biodiversity training data. So the *true* transfer
effect is **smaller** than the manuscript reports.

That makes the paper's own conclusion **safer, not weaker**. "Even with cross-dataset transfer and a
class-balanced sampler, gains are modest and error persists" survives a correction that shrinks one
of those gains — it is the direction the argument wants. A confound that had *deflated* the effect
would be the dangerous one; this is the opposite.

So: no design flaw, no re-run, nothing to re-measure. **Four sentences change**, and the argument they
support gets stronger.

---

## The one fact behind corrections 1–3

The two transfer cells train on the Biodiversity data **twice** (90 epochs); the other two train
once (45). That second pass is a full warm restart with a fresh optimiser and a fresh LR cycle.

**This project has already measured what that second pass alone is worth**, on 5 seeds, with no
transfer involved — `notes/SELFDISTIL_VERDICT_2026-06-22.md`:

| | baseline | +45-epoch warm restart | gain |
|---|---|---|---|
| mIoU | 80.60 | 82.04 | **+1.44** |
| Semi-natural IoU | 75.60 | 80.07 | **+4.47** |

It is quoted in a config that ships — `config/biodiversity/stage3_clsbal.py:88`:
> *"a +45-ep cycle alone buys ~+1.4 mIoU / +4.5 Semi-natural"*

So the extra pass is worth roughly what the transfer effect is worth. It is not a hypothetical
alternative; it is a measured one.

---

## Correction 1 — `main.tex:201` states something untrue

> **DONE 2026-07-27.** §2.3 now reads "All four cells share the same split, augmentation and
> evaluation protocol, but they are not matched on optimisation." Factor A renamed to
> *OpenEarthMap pre-training* throughout, including the Table 2 header and the interaction label.


**Currently:**
> "All four cells share identical optimisation settings, data splits, augmentation, and evaluation
> protocols, **differing only in the two factors under study**."

**Why it is wrong:** the cells differ in Biodiversity epochs (45 vs 90), in number of LR schedules
(one vs two), and in number of checkpoint selections (one vs two).

**Write instead:**
> "All four cells share the same data splits, augmentation and evaluation protocol. They are not
> matched on optimisation: the two transfer cells pass over the Biodiversity training set twice
> (90 epochs, two cosine cycles, two checkpoint selections) against once for the other two
> (45 epochs). Factor A is therefore the *pre-train-then-finetune procedure*, not cross-dataset
> transfer in isolation, and is reported as such."

## Correction 2 — `main.tex:433` eliminates two explanations and omits the live one

> **HALF DONE 2026-07-27.** The subtractive fix is applied: the mechanism attribution and the
> $+1.66$ pp are gone from §4.2. The replacement sentence needs a number from stage C1 → **PT2**.
> The only surviving `1.66` is inside Table 2, which is frozen withdrawn Results.


**Currently:**
> "cropland gains $+1.66$ pp from OEM transfer… **This gain cannot come from more cropland pixels or
> from rebalanced class counts, so it points instead to the broader, more transferable
> representations** learned from the diverse OpenEarthMap imagery."

**Be precise about WHICH clause is wrong.** "Pre-training adds no cropland examples at all" is
defensible: pool ∩ train = 1,072 and train − pool = 0, so the pool adds no cropland pixel the
baseline does not already train on. The defect is the next clause — *"cannot come from more cropland
pixels or from rebalanced class counts, **so it points instead to** the broader, more transferable
representations."* That is an exclusion argument naming two alternatives and omitting the third the
design itself creates: the same cropland pixels, seen twice.

**The fix available TODAY is SUBTRACTIVE. Do not write a replacement sentence yet.**
The $+1.66$ pp is from Table 2, computed on the withdrawn 219-tile validation split, and CLAUDE.md's
rule is that such numbers must never appear. A "corrected" sentence that carries $+1.66$ forward
repeats the exact failure being corrected. Worse, the paragraph's subject may not survive the new
split at all — cropland is among the thinnest classes on it.

So, now: **strike the mechanism attribution and the number.** Keep only what the design supports:
> "Cropland receives no external supervision: no OpenEarthMap class maps to it
> (Figure~\ref{fig:oemmap})."

After stage C1 produces a real number, and only if the gain reproduces, add:
> "It nonetheless gains [X] pp under factor A. Because the pre-training pool combines OpenEarthMap
> with the Biodiversity training tiles, that gain has two sources this design cannot separate — the
> broader representations from the diverse OpenEarthMap imagery, and the second pass over the
> Biodiversity cropland the pool already contains. We report the effect and do not attribute it."

## Correction 3 — `main.tex:322` makes the same move for semi-natural

> **OPEN → PT2.** The sentence sits in §3, which the pre-write brief froze. Note it will
> contradict the new §2 the moment §3 is unfrozen: §2 now states no OEM class maps to
> semi-natural grassland, while `main.tex:322` still says it is "mapped from bareland".


**Currently:**
> "The semi-natural grassland gain does not reflect substantial added exposure to that class… **It
> instead reflects OpenEarthMap's broader land-cover variety**, which gives a stronger, more general
> starting point…"

**Why it is wrong:** same structure, and semi-natural is the class where the measured second-pass
effect is largest (**+4.47**). This is the least defensible sentence in the paper.

**Same rule as Correction 2: subtractive now, rewritten after C1.** Strike "It instead reflects…"
and everything that attributes a mechanism. What survives today is the observation without the
explanation. The replacement sentence gets written once there is a number from the current split.

## Correction 4 — Methods must state the step counts

> **DONE 2026-07-27.** §2.3 states the 3,190-tile pool, 48,240 against 24,120 Biodiversity
> gradient steps, the ratio of exactly 2.00, and that it is declared rather than corrected.


Add to §2, from `docs/METHODS_STATED_LIMITATIONS.md` §1, which already has the arithmetic:

> "Stage 2a pre-trains on a pool of 3,190 tiles of which 1,072 are the Biodiversity training tiles
> themselves. Factor A therefore delivers 48,240 Biodiversity gradient steps against the
> baseline's 24,120, a ratio of exactly 2.00. This was identified before any model was trained and
> is declared rather than corrected; the step-matched control was not run."

## Correction 5 — free, and it is real evidence

> **OPEN → PT2.** Needs campaign output. §2 does now establish the reporting shape it depends on:
> two test sets, reported separately and never pooled.


Report factor A on **Test A and Test B side by side**. The second pass is over inland training
ground; OpenEarthMap's contribution is diversity. If factor A helps Test B (uplands, ~58 km away,
never surveyed) as much as Test A, the second-pass explanation weakens. If Test A greatly exceeds
Test B, it is live. Costs nothing, needs no threshold, and the campaign already produces it.

## Correction 6 — a scale caveat in a WORKING DOC, not the paper

> **DONE 2026-07-27.** The clause is in `METHODS_STATED_LIMITATIONS.md` §11.


**No new table goes in the manuscript.** This concerns the table already inside
`docs/METHODS_STATED_LIMITATIONS.md` §11, which is an internal working record that feeds the
write-up. The correction is one clause in that doc. §11's table of "two rate ratios in the literature" lists
Csurka (r = 5 px) and Volpi & Tuia (3 px erosion). I recomputed both derivations from the papers and
**§11's arithmetic is exactly right** — Vaihingen 1.24–1.33 and Potsdam 1.14–1.19 reproduce to the
decimal from Volpi's Tables I and III.

What the table does not say is the **scale**: Volpi's band is 3 px at 9 cm and 5 cm, i.e. **0.27 m
and 0.15 m**. Ours is **8 m** — 30x and 53x wider. §11 already warns the estimands differ; it should
also say the widths differ by an order of magnitude, or a reader will read 1.14–1.33 as a
same-scale benchmark for our rho.

**Add one clause:** "…and both are measured at 0.15–0.27 m, one to two orders of magnitude narrower
than the 8 m band used here, so neither is a benchmark for its value."

---

## Also verified while checking this

All of METHODS §10's and §11's **quotations are accurate**, now that the three papers have been
obtained and converted (`references_md/SOURCES_BOUNDARY_LITERATURE.md`). Kohli's *"error was computed
for different widths of the evaluation region"*, the *"8 pixel band"* in a figure caption, the
320x213 MSRC resolution, the *"quite rough"* ground truth; Cheng's *"not symmetric and favors
predictions whose masks are larger"* and the *"annotation consistency sets the lower bound on d"*;
Volpi's erosion sentence — all present, in context, as quoted. The §10/§11 readings were done
properly. They simply left no artefact anyone could check until now.

---

## Correction 7 — TWO different statistics are both called rho

> **OPEN → PT2.** Code, not manuscript. It blocks one manuscript item: §2 defines the 8 m band
> but not the 1.5 m contact zone that §3 and the appendix use, and documenting that zone before
> the rename would write the defect into the paper.


**Verified in the code 2026-07-27.** Two scripts in the same stage compute a boundary/interior ratio
from partitions that are not the same partition:

| script | "near" | "interior" |
|---|---|---|
| `boundary_rate_ratio.py` | within **8 m** | beyond 8 m |
| `boundary_trimap_iou.py:243` | within **1.5 m** | beyond 8 m |

`BND_MAX_M, INT_MIN_M = 1.5, 8.0`. So in the second, the **1.5–8 m annulus is in neither set** — and
on the current split that annulus is most of the band: 71.75% of the 8 m band lies beyond 1.5 m.

**Fix — the cheap one, and it is a rename, not new machinery.** They are genuinely two different
analyses and both are worth reporting: 8 m is the headline boundary/interior ratio, 1.5 m matches
Cheng's Boundary-IoU distance. Keep both, give the 1.5 m version its own name, and never call both
"rho". Under a working day, CPU only, nothing trained yet.

Also strike the stale framing while in there: `boundary_rate_ratio.py:3` opens *"rho — the
pre-registered primary statistic"* eight lines above the paragraph withdrawing that registration, and
calls the ensemble argmax "the registered estimator" at lines 272, 293 and 328.

## Correction 8 — the second arm is a necessary condition, not a diagnosis

> **ALREADY SATISFIED — verified 2026-07-27.** Both qualifiers are present in
> `METHODOLOGICAL_CHOICES.md` §E5 and in `CLAUDE.md` (an earlier check missed them because the
> phrases wrap across lines). `main.tex` keeps the compliant "convergent evidence rather than
> proof" sentence. The missing across-cell implementation is code → **PT2**.


The arm reads: if error is label-limited, the near-boundary rate stays flat across cells while the
interior rate falls. **Observing that does not establish the premise.** Every rival cause that is
constant across the four cells predicts the same pattern — encoder–decoder edge blur, mixed pixels at
0.5 m, image-to-vector registration offset — and the architecture is held constant in every contrast
the design computes, so none of them can be excluded here.

`main.tex:471` already gets this right (NOT 459 -- 459 carries the FORBIDDEN "binding constraint" sentence and must be struck): *"such models blur edges even on clean labels, so no single
measurement separates the two. We read this as convergent evidence rather than proof."* **Keep that
sentence.** The problem is the internal record, not the paper: `CLAUDE.md` and
`METHODOLOGICAL_CHOICES.md` §E5 state the arm without the two qualifiers the registered version
carried — the scale ("in relative terms") and the falsifier ("if both fall proportionally, the
concentration is a property of model quality"). Restore both.

**And note there is no implementation.** `boundary_trimap_iou.py` writes four separate per-cell JSONs
and `boundary_limited_error.py` renders one cell, so "roughly flat" would currently be adjudicated by
eye. Either compute the across-cell comparison or stop calling it an arm.

## Correction 9 — presentation of the audit trail (free, and worth more than it costs)

> **DONE 2026-07-27.** §2.6: "a threshold on that ratio was registered and withdrawn on
> 26 July 2026, before any model was trained on this split."


A leakage retraction guarantees a referee asks whether the second set of numbers was tuned after the
first was withdrawn. **This project has a dated answer** — every decision and all three
pre-registration versions predate any training on the corrected split — but it was filed behind the
wreckage. Done 2026-07-27: the audit trail moved to `docs/audit/` with a framing README,
`METHODOLOGICAL_CHOICES.md` and `DO_NOT_ADD.md` tracked, and `README.md` / `RUNBOOK.md` /
`DESIGN_NOTES.md` bannered as describing the withdrawn split.

**Still to write, one sentence in §2:** that a threshold on rho was registered and withdrawn before
any model was trained on the corrected split, with the date. Volunteered it is a credit; discovered
it is a question.

## Correction 10 — the spatial split is uncited, and the tile counts are described as ground

> **MOSTLY DONE 2026-07-27.** New §2.1.2 *Spatially Blocked Split* carries (a) Roberts 2017 and
> Kattenborn 2022, both read in the conversions before citing and both added to `Bibliography.bib`;
> (b) which gap does which job; (c) ground in km² never summed tiles; (d) the 2.85× scoring
> multiplicity and that it cancels in every contrast; (e) re-cutting declined; (f) the split
> described plainly; (h) realised 1,664 m / 768 m against the 750 m range, and **650 m appears
> nowhere in the manuscript**; (k) the 5/8 support floors appear **nowhere**; (l) the Test B hedge.
> **Open → PT2:** (g) and (i), both post-campaign, and the two code one-liners in (n).
> (j) and (m) are standing rules with nothing to execute.


**Verified in the code and the data 2026-07-27.** Measured directly from the GeoTIFF bounds: every
tile is 512 x 512 px at 0.5 m, i.e. a 256 m footprint, and tile origins are spaced exactly 128 m
apart in both axes. The chipping was not done here — there is no tile-writing code anywhere in this
repo, and the imagery was delivered by ODOS (`README.md:84`). No reason for the 50% stride is
recorded, so the paper must describe it, not justify it.

**(a) The split has no citation, and it is the most distinctive thing in the methods.** Nothing in
`manuscript/Bibliography.bib` supports spatially blocked validation. Three converted papers in
`papers-md/` cover it and none is cited (that directory is a private sibling checkout, so the references below are given by name rather than linked):

| paper | what it gives us |
|---|---|
| Roberts et al. 2017, *Ecography* | the canonical statement that random CV on structured data causes *"serious underestimation of predictive error"* |
| Kattenborn et al. 2022, *ISPRS Open J.* | the same result for **CNNs on remote sensing imagery**, random vs spatially blocked hold-outs — our exact case |

Both are converted and linked above; read the conversion before quoting either.
Wadoux et al. 2021,
*"Spatial cross-validation is not the right way to evaluate map accuracy"*, was considered and does
**not** apply: it disputes spatial CV as a way to estimate map accuracy over a target population,
which is not claimed here, and says nothing about held-out tiles containing the same pixels as
training tiles. Do not cite it. Recorded so nobody spends an afternoon rediscovering it.

**And the design does meet the standard those two papers set, on the split that carries the result.**
Measured separations, from the manifest and recomputed from the GeoTIFF bounds:

| pair | separation | inland composition range 750 m | inland spectral range 1,350 m |
|---|---|---|---|
| train — test | **1,664 m** | clears | clears |
| val — test | 768 m | clears | does not |
| train — val | **256 m** | does not | does not |

Test A is separated from training ground by 1,664 m, beyond the distance at which the landscape stops
resembling itself on either measure — the composition correlogram's first non-significant bin is
700–800 m (Mantel r = 0.0024, Holm p = 0.15). The known exception is validation, which sits 256 m from
training and is where every checkpoint is selected; that is already in the stated-properties table
above and must not be quietly dropped when the citations go in.

**(b) Two reasons for a spatial split, which the current text runs together.** Repeated pixels — a
held-out tile physically overlapping a training tile, which is recall rather than prediction — and
spatial autocorrelation, where nearby but non-overlapping ground is still the same fields and
hedgerows. The 256 m val buffer addresses the first, the 768 m test gap the second (inland
composition range 750 m). Say which gap is doing which job.

**(c) Tile counts are being written as though they were ground.** Exact union of the 294 Test A
footprints is **6.767 km²**; adding the tiles up gives 19.268 km², which is 2.85x the same ground.
Never report the summed figure. Corrected in `utils.py:201`, `boundary_rate_ratio.py:48` and
`METHODS_STATED_LIMITATIONS.md:178` on 2026-07-27, where 6.783 km² / 7.52 cells did not reproduce and
are now 6.767 / 7.50.

**(d) Scoring double-counts, and nothing says so.** `Evaluator.add_batch` sums one global confusion
matrix over tiles with no deduplication, so a patch appearing in four tiles contributes four times.
Across Test A's 413 distinct 128 m cells the multiplicity is 41 / 118 / 117 / 137 cells at 1 / 2 / 3 /
4 times, mean 2.85. Strip-interior ground therefore carries roughly four times the weight of
strip-edge ground in every reported rate. **This does not touch any comparison** — all four cells are
scored on identical pixels with identical weights, so it cancels exactly in every contrast — but it
does affect absolute levels. State it, or deduplicate before scoring.

**(e) Re-cutting without overlap: available, declined, and it should be said so.** Reassembly is
exact rather than estimated (tiles sit on a perfect 128 m grid and overlap regions are byte-identical,
`METHODS_STATED_LIMITATIONS.md:370`). Re-cutting the training ground at a 256 m stride would drop
1,072 training tiles to about **355** — 23.28 km² of ground at 0.0655 km² per disjoint tile — while
removing no ground at all. It would also invalidate the 45-epoch schedule, which restarts its
learning rate at 15 and 45. And it would remove only the 256 m val buffer: the geographic cut, the
768 m test gap, the dropped straddling tiles and the whole-site upland hold-out all exist for
autocorrelation and would survive unchanged. Declined on that basis.

**(g) POST-CAMPAIGN, cannot be run before results exist.** The train|val gap is 256 m, so validation
is mildly optimistic, and the argument that this is harmless rests on the optimism being common-mode
across the four cells. It is not quite: the transfer cells select a checkpoint twice and the other two
select once (B4), so the transfer arm gets two draws on that optimism. Once the campaign returns,
compare the validation-minus-test gap per cell. If it is larger for the transfer cells, factor A is
inflated — the same direction as the 2.00x step confound. Minutes to compute; report either way.

**(h) The 650 m buffer is justified by the wrong quantity. Report the realised separations instead.**
`build_spatial_split.py:523` justifies it as *"650 m = 2.5x the 256 m tile footprint"* — geometry, not
autocorrelation — and 650 m is also **ireland1's** composition range, a different site from the one
being cut. The inland site's range is **750 m**, so the requested buffer is below it. The split is
nonetheless sound: quantised onto the 128 m grid the realised separations are **1,664 m train—test**
and **768 m val—test**, both above 750 m. Good outcome, wrong reason. Report the realised separations
against the inland 750 m range and drop 650 m from the argument entirely.

**(i) Test B pools two sites that are not twins, and nothing reports them separately.**
`RUNBOOK.sh:504` scores `external_test` as one directory, and no script in `evaluation/` breaks it
down by site. But ireland1 is 64 tiles on 1.82 km² with a 650 m composition range and ireland2 is 127
tiles on 3.34 km² with a 950 m range, ~58 km apart, so the pooled figure is roughly two-thirds
ireland2. **This is a missed opportunity rather than a defect** — the pooled number is not wrong, it
just hides its own fragility. Report both sites separately: it is the same predictions counted in two
piles, costs no compute, and if the two disagree that gap *demonstrates* the n = 2 limit the prose
currently only asserts. If they agree it is a free robustness point.

**This is a check to run, not a result to report — default is zero extra numbers in the paper.**
Compute the per-site split of the headline Test B mIoU once and look at it. If the two sites agree,
write one clause and no numbers: *"the two upland sites agree closely."* Only if they diverge does it
earn a parenthetical, and in that case the divergence is itself the finding and you want to have seen
it before a referee does. No new table, no new section, no per-class breakdown.

**(j) Do NOT write that the 45-epoch budget was set for a larger training set and left unrevised.**
It is true that steps fell — 1,553 inland tiles at 45 epochs was 34,920 updates, 1,072 is 24,120, a
31% drop — but that is because there is less ground, not because the schedule is now short. Measured:
the old training strip held 40.42 km² at multiplicity 2.52, the new one 23.28 km² at 3.02, so 45
epochs now exposes each patch of ground **136 times against the old 113**. Per unit of ground the
model trains 20% *harder* than the tuned configuration did. The schedule stays at 45 (two complete
CosineAnnealingWarmRestarts cycles, T_0=15, T_mult=2; the next aligned stop is 105) and the learning
rate is not touched. The live risk is overfitting rather than undertraining, and val-best checkpoint
selection already handles it.

**(k) Every design constant, classified — and the split satisfies the published rule.**
Checked against the two converted papers rather than asserted. Roberts states two different rules and
the applicable one is line 209, on **raw data**: *"at least as many units as the range of
autocorrelation… dependence structures in this step are assessed on raw data."* (Line 92's
*"substantially larger"* concerns **residual** autocorrelation, which is not what our correlogram
measures.) Roberts line 92 also gives the buffer rule: *"a buffer size equivalent to distances at
which residual autocorrelation is reduced to zero suffices."* Realised train—test separation is
**1,664 m against a 750 m range** — the published rule is met with room, and that is the sentence to
write.

| constant | provenance | arbitrary? |
|---|---|---|
| 256 m val buffer | 512 px x 0.5 m, the exact pixel-identity distance | no, exact |
| 45 epochs | two complete CosineAnnealingWarmRestarts cycles | no, derived |
| 950 m support cell | ireland2's measured composition range | measured but from the wrong site; >= 750 m, so it satisfies Roberts regardless |
| 1.5 m contact zone | Cheng's Boundary-IoU distance | borrowed, defensible |
| 80/10/10 target | convention | arbitrary, universal |
| 650 m requested buffer | 2.5x the tile footprint | **yes** — superseded by the realised 1,664 m, see (h) |
| 5 / 8 class-support floors | nothing | **yes, and no precedent exists in either cited paper** |
| 8 m boundary band | nothing | **yes** — already declared a-priori, METHODOLOGICAL_CHOICES E1 |

**The 5/8 floors must not appear in the paper.** They were a filter on candidate cuts during the
15,000-restart search, applied before anything was trained, so they cannot flatter a result — but no
published method has such a rule, and 8 is not defensible if a referee asks. `block_phase_sweep.json`
shows the shipped grid anchor clears the test floor by exactly zero (8 against 8) and that five of ten
equally valid anchors would have failed it, which is a property of the anchor, not of the land.
Describe the split that exists and its measured separation. Do not describe the search that found it.

**What the sweep IS good for, and it is a strength rather than a confession.** Roberts line 199
recommends exactly it: *"cross-validations could be run several times with spatial or other structured
blocks defined in a variety of sizes and/or orientations. This approach produces a range of validation
statistics… rather than just a single value."* `block_phase_sweep.json` and
`block_size_sensitivity.json` are that. One clause, citing Roberts, and it is done.

**(l) Report Test A per class; report Test B as one number, with one sentence that cannot be dropped.**
Measured grid-cell support at 950 m: Test B's Cropland occupies **4** cells (30 tiles) and Settlement
**6**, against **8** and 13 on Test A. (Corrected 2026-07-27: this line said 16 for Test A's Cropland;
`artifacts/class_support.json` gives 8. 16 is Test A's Forest/Grassland/Semi-natural count.) Drop the per-class Test B breakdown — those two rows would each
need their own hedge and the paper does not need the claim.

**But note what does NOT follow, because the first version of this correction got it wrong.** Reporting
only the overall figure does not retire the problem. Foreground mIoU is the unweighted mean of the five
per-class scores (METHODOLOGICAL_CHOICES F2), so Cropland-on-4-cells is *inside* the headline number
whether or not it is printed. Burying it is worse than stating it. The hedge cannot be removed, only
made small — one sentence carrying both facts:

> Test B is a different landscape — semi-natural is 60.3% of its foreground against 4.3% in training —
> and two of its five classes occupy very little ground, so its mIoU is reported as a check on transfer
> rather than as an accuracy estimate.

**Do not pool Test A and Test B** — averaging a surveyed-area result with an unsurveyed-terrain result
answers neither question and conceals that they are different places.

**(m) The factorial needs an UPPER BOUND, not a decomposition — so it needs far fewer statistics.**
The contribution is the diagnosis, not the 2x2 (METHODOLOGICAL_CHOICES A3). The factorial exists only
to establish that the two interventions give modest gains, so that something else must be binding.

The 2.00x step confound **inflates** factor A: part of what is credited to transfer is a second pass
over the Biodiversity training data. A confound that makes a gain look *larger* cannot threaten a claim
that the gain is *small*. The confounded estimate is therefore an upper bound, and an upper bound is
exactly what the argument needs — *even crediting the procedure with everything, the gain is modest.*

**Consequence, and it is a simplification.** No mechanism attribution (Corrections 2 and 3 already
strike those sentences), no step-matched control, no decomposition of A into transfer versus extra
pass, no interaction-term interpretation beyond reporting it. Four cells, foreground mIoU, paired
per-seed differences, and the per-class table on Test A. Nothing further belongs in the factorial
section, and adding more would move weight onto the part of the paper that is not the contribution.

**(n) Interrogation elements 10–20, checked in the code 2026-07-27. No design change required.**
Each was verified by reading the shipped file, not from memory or from these docs.

| element | checked | verdict |
|---|---|---|
| 10 interaction term | `aggregate_seeds.py:100`, textbook Montgomery contrast | report it, do not interpret it; **rename the label** |
| 11 "modest gains" | see (m) | supported as an upper bound |
| 12 the 8 m band | `BAND_M = 8.0`, one constant, one use site, self-tested | consistent; already declared a-priori (E1) |
| 14 GT-only banding asymmetry | METHODOLOGICAL_CHOICES E2 and METHODS §10/11 both carry Cheng's *"not symmetric and favors predictions whose masks are larger"* | already stated |
| 15 boundary-free tiles | `boundary_rate_ratio.py:211` — a single-class tile *"must be EXCLUDED, not scored as zero"*, with a self-test that observes the exclusion and a null control | implemented and observed to work |
| 17 the exclusion curve | `boundary_trimap_iou.py:193-202` — per-seed curves over 10 seeds with SD, not the ensemble argmax | matches E4; the curve is the deliverable |
| 18 no spatial interval | see D1; the block bootstrap was removed 2026-07-26 | settled |
| 19 paired per-seed contrasts | `FG_CLASSES` all five plus `SCALAR_METRICS` mIoU/mF1/OA across all three splits | the D2 defect (three classes only) is fixed |
| 20 what Test B supports | see (l) and DO_NOT_ADD | settled |

**Two small items fall out, both one-liners.**

1. `aggregate_seeds.py:87` labels the interaction `"transfer x sampler (interaction)"`. Factor A is the
   pre-train-then-finetune **procedure**, not transfer — Correction 1 fixed exactly this naming for the
   main effect and left the interaction untouched. Read *procedure x sampler*.
2. `SPLIT_DIRS` aggregates **val** alongside test and external_test. That is fine as a convergence
   check, but validation sits 256 m from training and is where every checkpoint is selected, so no
   headline number may be drawn from it. Report it, never lead on it.

**(f) The split in four sentences, because §2 currently describes it inconsistently.**

> Tiles were delivered on a grid with 50% overlap, so a random tile split would place the same ground
> in both training and test. We instead cut one site into three geographic strips — train, validation
> and test — separated by gaps wide enough that the landscape on either side is no longer correlated.
> Tiles falling inside a gap are discarded. Two upland sites are held out whole as a second, harder
> test set.
