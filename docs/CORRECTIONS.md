# CORRECTIONS — verified, and they need to happen

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

Add to §2, from `docs/METHODS_STATED_LIMITATIONS.md` §1, which already has the arithmetic:

> "Stage 2a pre-trains on a pool of 3,190 tiles of which 1,072 are the Biodiversity training tiles
> themselves. Factor A therefore delivers 48,240 Biodiversity gradient steps against the
> baseline's 24,120, a ratio of exactly 2.00. This was identified before any model was trained and
> is declared rather than corrected; the step-matched control was not run."

## Correction 5 — free, and it is real evidence

Report factor A on **Test A and Test B side by side**. The second pass is over inland training
ground; OpenEarthMap's contribution is diversity. If factor A helps Test B (uplands, ~58 km away,
never surveyed) as much as Test A, the second-pass explanation weakens. If Test A greatly exceeds
Test B, it is live. Costs nothing, needs no threshold, and the campaign already produces it.

## Correction 6 — a scale caveat in a WORKING DOC, not the paper

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

The arm reads: if error is label-limited, the near-boundary rate stays flat across cells while the
interior rate falls. **Observing that does not establish the premise.** Every rival cause that is
constant across the four cells predicts the same pattern — encoder–decoder edge blur, mixed pixels at
0.5 m, image-to-vector registration offset — and the architecture is held constant in every contrast
the design computes, so none of them can be excluded here.

`main.tex:459` already gets this right: *"such models blur edges even on clean labels, so no single
measurement separates the two. We read this as convergent evidence rather than proof."* **Keep that
sentence.** The problem is the internal record, not the paper: `CLAUDE.md` and
`METHODOLOGICAL_CHOICES.md` §E5 state the arm without the two qualifiers the registered version
carried — the scale ("in relative terms") and the falsifier ("if both fall proportionally, the
concentration is a property of model quality"). Restore both.

**And note there is no implementation.** `boundary_trimap_iou.py` writes four separate per-cell JSONs
and `boundary_limited_error.py` renders one cell, so "roughly flat" would currently be adjudicated by
eye. Either compute the across-cell comparison or stop calling it an arm.

## Correction 9 — presentation of the audit trail (free, and worth more than it costs)

A leakage retraction guarantees a referee asks whether the second set of numbers was tuned after the
first was withdrawn. **This project has a dated answer** — every decision and all three
pre-registration versions predate any training on the corrected split — but it was filed behind the
wreckage. Done 2026-07-27: the audit trail moved to `docs/audit/` with a framing README,
`METHODOLOGICAL_CHOICES.md` and `DO_NOT_ADD.md` tracked, and `README.md` / `RUNBOOK.md` /
`DESIGN_NOTES.md` bannered as describing the withdrawn split.

**Still to write, one sentence in §2:** that a threshold on rho was registered and withdrawn before
any model was trained on the corrected split, with the date. Volunteered it is a credit; discovered
it is a question.
