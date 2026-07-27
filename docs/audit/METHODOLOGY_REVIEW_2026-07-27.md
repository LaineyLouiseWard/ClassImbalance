# Independent methodology review — 2026-07-27

Answers `docs/audit/BRIEF_METHODOLOGY_REVIEW.md`. Reviews the science, not the code: whether the experiment
as built can support the claims as written, and where it cannot, whether the fix is the design, the
analysis, or the sentence.

**Sources read in full:** `docs/METHODOLOGICAL_CHOICES.md`, `manuscript/main.tex`,
`docs/audit/DECISIONS_REBUILD_2026-07.md`, `docs/METHODS_STATED_LIMITATIONS.md`, `notes/PAPER_PURPOSE.md`,
`CLAUDE.md`, `notes/CITATION_INTENT_2026-07-25.md`, `references_md/SOURCES_BOUNDARY_LITERATURE.md`,
and the analysis code named below. All four boundary-evaluation sources were opened and their
quotations checked (§5).

**Nothing was changed.** No edits, no commits, no runs.

---

## 1. Can the paper make its central claim?

**Yes with qualification — but the qualification is larger than the internal documents assume.** The
design supports *"the residual error is boundary-localised and neither data-level intervention
reaches it."* It does not identify **label quality** as the cause, because three rivals produce the
same signature and none is separable here: mixed pixels at 0.5 m, boundary blur from the
encoder–decoder, and any sub-metre registration offset between the Pléiades raster and the QGIS
digitisation. The fixed-architecture constraint makes the architecture rival *permanently*
unfalsifiable, since architecture is a constant in every contrast the design computes. The
manuscript already concedes this at `main.tex:459`; `METHODOLOGICAL_CHOICES.md` E5 and CLAUDE.md's
"second arm" do not, and the §2 rewrite is at live risk of importing the stronger internal wording
into a paper that currently has the weaker, correct one. Shortest statement of what must change: the
claim's scope moves from *labels are the binding constraint* to *the residual is boundary-limited and
label-free curation cannot reach it*, and the practical recommendation is conditioned — re-annotation
cannot fix a mixed pixel.

---

## 2. Findings

### 1. [ANALYSIS] The boundary diagnostic is computed on a ten-seed ensemble, and ensembling inflates the very quantity being claimed

<evidence>

`scripts/analysis/boundary_rate_ratio.py:271`

    return stack.mean(axis=0).argmax(axis=0)

`scripts/analysis/boundary_trimap_iou.py:137`

    ens_pred = stack.mean(axis=0).argmax(axis=0).astype(np.int64)

`scripts/analysis/boundary_trimap_iou.py:176`

    err_e_k[k] += np.bincount(bidx[sel_k & (ens_pred != mask)], minlength=nb)

`docs/audit/DECISIONS_REBUILD_2026-07.md:81` (D9, on retracting `lift`)

> "It also rises as a model improves, because a better model removes interior error first, so it
> partly tests how well the model was trained — the alternative hypothesis the claim exists to
> exclude."

</evidence>

**What is wrong:** rho and the per-class boundary/interior rates are the *ensemble's* residual, not
any factorial cell's. An ensemble is strictly better than its members and removes interior error
preferentially, so boundary concentration measured on it exceeds that of any model the paper
tabulates. This is the identical mechanism used to retract `lift` — applied to `lift` it was
disqualifying; applied to rho it is unremarked. Second, it collapses ten seeds into one number, so
the study's single declared estimator (`METHODS_STATED_LIMITATIONS.md` §7, *"Uncertainty is PER-SEED
AND PAIRED… One estimator, not two"*) does not cover the headline diagnostic. As coded it has no
estimator at all. Note the asymmetry inside one file: A1 and A3 accumulate per seed
(`boundary_trimap_iou.py:138`, `seed_preds`), A2 does not.

**What it costs the paper if unfixed:** a reviewer asks why the central number comes from a system
that appears in no table, and whether the concentration survives on a single model. There is no
answer, because it was never computed.

**Smallest fix:** `seed_preds` already exists and is already used twice in the same loop. Extend the
A2 accumulators (`err_n`, `err_e`, `err_n_k`, `err_e_k`) over the seed axis; report per-seed
mean ± SD and keep the ensemble as a secondary row. ~15 lines, CPU only over dumps stage C5 produces
anyway. **Zero GPU. Fits in two days — write it today, while the campaign runs, so it costs nothing
on the critical path.**

---

### 2. [ANALYSIS] On Test B the "boundary" set is contaminated by the survey edge, and the robustness check that would catch it has no code path

<evidence>

`scripts/analysis/seed_disagreement.py:114-118` — boundaries are computed on the **full** mask, so
every foreground/background contour counts:

    bnd[:-1, :] |= m[:-1, :] != m[1:, :]
    bnd[1:, :]  |= m[:-1, :] != m[1:, :]
    bnd[:, :-1] |= m[:, :-1] != m[:, 1:]
    bnd[:, 1:]  |= m[:, :-1] != m[:, 1:]

`CLAUDE.md`

> "Background is 1.7% of training pixels but 38% of Test B."

`docs/audit/DECISIONS_REBUILD_2026-07.md:165` (D13)

> "The NaN mask … always touches a tile border, averaging 42% of a Test B tile — that is off-mosaic
> fill."

`manuscript/main.tex:372`

> "Restricting it to contacts between two different foreground classes, excluding background and
> image-edge contours, leaves the within-8~m error share essentially unchanged (92\% validation,
> 96\% test), as does using eight-connectivity."

</evidence>

**What is wrong:** on Test B roughly 38% of pixels are background, mostly off-mosaic fill at tile
borders. Every foreground pixel near that fill is scored as *within 8 m of a class boundary* — but it
is the edge of the flown survey, not a land-cover boundary. Error there is independently elevated:
D14 records the encoder sees dark regions through its receptive field in 80% of Test B tiles against
8% of training tiles. Both effects push rho up on the set that carries the transfer claim. The
manuscript asserts the check was done — but it was done on the withdrawn campaign, on a split where
background was 1.7%, and **no script in the tree implements a foreground-only boundary variant**
(grep across `scripts/` returns nothing). The one place the check matters is the one place it has
never been run.

**What it costs the paper if unfixed:** the Test B boundary numbers are unsound, and `main.tex:372`
states a robustness result that cannot be reproduced from the repository.

**Smallest fix:** add a `--foreground-boundaries-only` mode that computes `bnd` with background
pixels excluded from the neighbour comparison, and report both variants for Test A and Test B. ~10
lines in `boundary_distance`, plus a flag. CPU only, same dumps. **Fits in two days; same scheduling
as finding 1.**

---

### 3. [WRITE-UP] The second arm does not separate label-limited from capacity-limited

<evidence>

`CLAUDE.md`

> "Second arm, free: if error is label-limited, the near-boundary rate must be roughly flat across
> the four cells while the interior rate falls. rho alone cannot separate label-limited from
> capacity-limited, because rho rises as a model improves."

`docs/METHODOLOGICAL_CHOICES.md` A1

> "FT-UNetFormer is the industrial partner's deployed model. We never change it."

</evidence>

**What is wrong:** the four cells differ only in data. An architectural boundary limit — or a
registration offset, or mixed pixels — is a constant in that contrast, so flatness across cells is
predicted equally by every one of them. The arm shows the residual is not reachable by *these two
interventions*. It does not identify a cause. The paper's own text already gets this right; the
internal record does not, and the record is what §2 will be rewritten from.

**Smallest fix:** WRITE-UP, zero cost. Preserve `main.tex:459` verbatim through the rewrite, and
correct E5 / CLAUDE.md so the rewrite cannot import the stronger version.

**Replacement sentence** (for the internal record, E5 and CLAUDE.md):

> If error is label-limited, the near-boundary rate should stay roughly flat across the four cells
> while the interior rate falls. Flatness rules out class exposure and cross-dataset representation
> as the binding constraint. It does not distinguish label ambiguity from any other cause held
> constant by the fixed architecture, and the architecture is held constant by design.

---

### 4. [WRITE-UP] A large rho is close to arithmetically guaranteed; the informative quantity is how far the elevation extends

<evidence>

`manuscript/main.tex:368`

> "from $42\%$ in the half-metre band at the boundary to about $0.5\%$ in the deep interior beyond
> eight metres"

</evidence>

**What is wrong:** a model whose only defect is a one-pixel displacement of every contour has roughly
50% error in the half-metre band, zero interior error, and unbounded rho — **on perfect labels**. So
42% is what a sub-pixel-to-one-pixel offset produces, and neither the level nor the ratio carries
information about label quality. What does carry information is the *extent* of the decay: error
still above the floor at 4–6 m is 8–12 px and cannot be a one-pixel offset of an otherwise correct
contour. The paper already computes this (`error_vs_distance`, per-distance-bin rates) and leads on
the ratio instead.

**Smallest fix:** WRITE-UP plus a reordering of an existing curve. **No new statistic, and explicitly
no bar** — report the decay, do not invent a cut-off for "far enough".

**Replacement sentence:**

> The elevated error rate is not confined to the pixels immediately adjacent to a boundary. It
> remains above the interior floor out to *N* m, which is *2N* pixels at 0.5 m ground sample
> distance, and therefore cannot be accounted for by a sub-pixel or single-pixel displacement of an
> otherwise correct contour.

---

### 5. [WRITE-UP] Two a-priori boundary widths are used, one is undeclared, and they are not complementary

<evidence>

`docs/METHODOLOGICAL_CHOICES.md` E1

> "A band around every ground-truth class boundary, 8 metres wide… 8 m is an a-priori choice, stated
> as such."

`scripts/analysis/boundary_trimap_iou.py:229`

    BND_MAX_M, INT_MIN_M = 1.5, 8.0

`manuscript/main.tex:370`

> "comparing each class's own boundary band (within $1.5$~m) against its own deep interior (beyond
> $8$~m)"

</evidence>

**What is wrong:** 1.5 m is a second a-priori width and E1 declares only one. Worse, the 1.5–8 m
annulus falls in neither category, so the per-class contrast partitions the foreground differently
from rho (`< 8.0 m` vs `>= 8.0 m`). A reader who tries to reconstruct rho from the per-class panel
will not reproduce it, and the discrepancy is a discarded annulus that holds a large share of both
the foreground and the error.

**Smallest fix:** WRITE-UP, two sentences.

**Replacement sentence:**

> Two boundary widths are used and both are a-priori. An 8 m band partitions the foreground into
> near-boundary and interior sets for the pooled rate ratio; a narrower 1.5 m band is compared
> against the deep interior beyond 8 m for the per-class contrast. The two partitions are not
> complementary — pixels between 1.5 and 8 m enter neither — so the per-class contrast is not the
> per-class decomposition of the pooled ratio.

---

### 6. [WRITE-UP] No form of "generalises" is available for Test B

<evidence>

`docs/METHODS_STATED_LIMITATIONS.md` §6

> "external_test (Test B) | 191 | 5.164 km²"

`artifacts/class_support.json` — `external_test` Seminatural share 0.6033 against train 0.0426;
Cropland 4 blocks / 30 tiles; Settlement 6 blocks / 75 tiles.

`RUNBOOK.sh:424`

> "[C1b] Test B — the held-out upland sites. This is the generalisation number the paper leads on,"

</evidence>

**What is wrong:** two purposive sites, 5.16 km², between-site n = 2. Semi-natural is **60.3%** of
Test B foreground against 4.3% in training — a 14× prior shift — so any Test B result is a joint
statement about geographic transfer and composition, and the design separates neither. Foreground
mIoU weights Cropland (4.0% of foreground, 4 grid cells) and Settlement (1.5%, 6 cells) equally with
Semi-natural (60.3%). Test B is also structurally less boundary-dense than the inland site (band area
share 26.5% vs 37.8%), so it is a different landscape for the boundary claim, not a replicate of it.

**Smallest fix:** WRITE-UP. Change the word everywhere, including the RUNBOOK banner.

**Replacement sentence:**

> Test B is a complete enumeration of two upland sites totalling 5.16 km², held out whole. It is not
> a sample of Irish uplands, and its class composition differs sharply from the training site —
> semi-natural grassland is 60% of its foreground against 4% in training. A Test B result therefore
> states how this model performs on this ground; it is not an estimate of performance on unsurveyed
> terrain in general.

---

## 3. The six open questions

**1. B3 — declare the extra Biodiversity pass, or buy the clean comparison? Declare.** Not because
the control is infeasible: a second Biodiversity-only 45-epoch stage warm-started from each baseline
checkpoint is step-matched, costs ~1/3 of stage 2a (536 steps/epoch against 1,595), and as a separate
10-task array after the campaign is ~+2 h wall clock — it *would* fit inside two days. Decline it on
scope. It adds a fifth cell to a 2×2, which costs a paragraph of explanation in a page-limited
methods for a magnitude that is setup, not the contribution; and run inside the existing task it eats
the margin against `--time=20:00:00` in `sonic/campaign/campaign.slurm` at 12–16 h per task. But
D12's disclosure is defensive where it should be assertive. **Add the argument D12 does not make:**
the confound inflates main effect A, so it biases *against* the paper's own conclusion. If part of
the transfer gain is really a second pass over the training set, the genuine curation effect is
smaller than reported and the ceiling argument is stronger, not weaker. Say that in one sentence and
the confound stops being a liability.

**2. B2/B4 — must main effect A be renamed? Yes, everywhere, including the table header.**
`METHODS_STATED_LIMITATIONS.md` §1 measures the ratio at exactly 2.00×, exact by construction.
"Cross-dataset transfer" as the *name* of the contrast is then a claim the design cannot support, and
it currently appears in the abstract, the highlights, the conclusions and the `\textbf{OEM transfer}`
column header. Define it once in the methods as *OpenEarthMap pre-training, which also delivers a
second pass over the training set*, then use a neutral label — "OpenEarthMap pre-training" — as the
name thereafter. Do not let "transfer" stand alone in any results sentence.

**3. C4 — is it defensible that Test B faces no support floor? Yes, and the floor is the wrong thing
to worry about.** Test B is held out whole and was never a candidate to be re-cut, so a floor could
only have rejected the site. Report the support beside every Test B per-class number and move on.
**The real exposure is aggregation, not admission:** foreground mIoU gives Cropland (4 cells, 30
tiles) and Settlement (6 cells, 75 tiles) the same weight as Semi-natural at 60% of the foreground.
So report Test B per class with its support, and if an aggregate is quoted at all, quote it with the
composition beside it. Do not lead on a Test B mIoU.

**4. D1 — is "no spatial interval anywhere" right? Yes, and the reasoning holds — but the
presentation will read as certainty and that is fixable.** The census argument is correct for the
estimand as defined, and §7 reason 4 is honest that it does not answer the reader's actual question.
The failure mode is elsewhere: a per-seed interval printed next to a *level* is not an interval on
the level, it is training-run variation at fixed ground, and `89.1 ± 0.3` at `main.tex:360` will be
read as a precision claim about the number. §7 already states the rule — *"Levels are reported with
their spatial extent; contrasts carry the interval"* — and the draft violates it in two tables.
Enforce it: bare levels with their km² and tile count, intervals on contrasts only. That is a
formatting change, and it converts the absence of a spatial interval from something the reader must
be argued into, into something the layout makes obvious.

**5. E2 — does ground-truth-only banding bias the cross-cell comparison, or only the level? Only the
level — and also the per-class ordering, which the question does not ask about and which the paper
does use.** The band is built from the mask alone (`boundary_rate_ratio.py:101`, verified in code),
so `n_near` and `n_far` are bit-identical across the four cells and Cheng's asymmetry is a common
term that cancels in the difference. It does not cancel in the pooled level, which is optimistic for
a model that over-predicts. **And it does not cancel across classes**: a class the model
systematically over-predicts is flattered relative to one it under-predicts, so the per-class ranking
is biased — and the per-class ranking is what carries the "difficulty tracks boundary exposure"
argument at `main.tex:374`. State the asymmetry against the level and against the per-class ordering.
Do not state it against the contrast; that would concede something untrue.

**6. A2/A3 — what evidence for the label premise beyond ~88%? Almost none, and the paper does not
need any.** The ~88% justifies a *constraint* (re-annotation is infeasible), not the *claim*. It has
no recorded protocol — no inspector count, no tile count, no error criterion — it appears nowhere in
the manuscript, and the same number appears throughout `notes/` as an "inter-annotator bound"
attributed to "Ortiz et al. 2025 (TGRS)", **a source `docs/DO_NOT_ADD.md` records as fabricated**.
Two different 88%s in one notes directory, one of them with no real source behind it, is a conflation
waiting to happen — and per `CLAUDE.md` one external reviewer has already made it. Keep it out. The other candidates are not independent: confident
learning scores against the same labels and flags boundary pixels for the same geometric reason, and
Kohli's 2009 statement of the same premise on MSRC is a precedent, not evidence about this dataset.
The correct response is not to find more evidence — it is to narrow the claim, per §1.

---

## 4. The seventh thing

**The boundary instrument has never been run against an input that should make it fail — which is
precisely the failure mode this repository documents as its own recurring defect.**

`boundary_rate_ratio.py:147` has a good self-test: it plants rho = 6.0 and recovers it under two
landscapes whose band area shares differ by 2×, and it separately checks the 8 m width, strict
membership, anisotropy and the boundary-free exclusion. Every one of those tests confirms the
statistic **measures what it says**. Not one tests whether a large value **means what the paper says
it means**. The missing case is a null: clean labels, a prediction displaced by one pixel, and the
confirmation that rho comes out large anyway. It would come out large — and the test would fail in
the useful direction, by demonstrating that rho alone licenses no label conclusion.

CLAUDE.md states the rule and applies it only to gates: *"A gate that has not been observed to fail
does not exist. Construct the input it should reject and watch it reject it."* Applied to a
measurement, the same rule says an instrument whose output has never been generated by the
alternative hypothesis cannot discriminate against it. Six gates could not fail; rho's self-test
cannot either. That is the same defect at a different altitude, and it is the one thing on this list
not already written down somewhere in the repository.

Cost: one extra case in an existing self-test, half an hour. It changes no result. What it changes is
which sentence the paper leads with — from a ratio, to the decay extent of finding 4.

---

## 5. What held up

Choices attacked that survived:

- **The 768 m realised val/test separation against a 950 m autocorrelation figure.** Expected to
  fail; does not. The reader-facing guarantee is the 1,664 m realised train–test separation in the
  manifest, validation is never reported, and a peak Mantel r of 0.044 makes the residual dependence
  negligible in effect size, not merely arguable. D5's "justify from the realised separation, not the
  correlogram" is the right defence.
- **The 950 m block provenance.** It is another site's number, and the split's admissibility is
  contingent on it (it fails at 1,350 m). §4's argument survives: 950 m sits *above* the inland
  composition range of 750 m, so it counts fewer independent units than the criterion's own scale and
  cannot flatter the support; 1,350 m is the spectral range and answers a different question. No
  reading was found in which the 1,350 m failure bites.
- **Removing the block bootstrap.** Attempted argument: a reviewer will demand one anyway. Reason 3
  of §7 is decisive and was verified in code — the band pixel sets are built from the mask alone, so
  they are bit-identical across cells and the landscape genuinely is a constant in every contrast.
  Reinstating it would be apparatus for a question the paper does not ask.
- **D11, grounded argmax.** The audit's decisive reason was checked against §3's measured table. The
  OEM half contributes 0.000% of both Cropland and Semi-natural; the Biodiversity half contributes
  7.703% and 4.265%. Both channels receive positive evidence throughout stage 2a. D11's refutation is
  correct and the override should stay rejected.
- **§10 and §11's readings of the boundary literature.** All four sources are now converted on this
  machine (`papers-md/kohli-2009-*`, `cheng-2021-*`, `volpi-2017-*`, converted 2026-07-27 per
  `references_md/SOURCES_BOUNDARY_LITERATURE.md`). **Every load-bearing quotation checks out
  verbatim.** Kohli's "8" is 8 pixels in a figure caption and the refusal to cite it for 8 m is
  right; "trimap" occurs exactly three times, all in Fig. 17; Cheng's asymmetry sentences are at
  lines 59 and 146 of the conversion and the annotation-consistency rule at 206; Csurka's
  trimap-limitation passage and the validation-set footnote are at lines 75 and 124. The derived
  Volpi & Tuia ratios were recomputed from Tables I and III: Vaihingen 1.235–1.329 and Potsdam
  1.135–1.187, which round to the **1.24–1.33** and **1.14–1.19** §11 reports. §11's arithmetic is
  correct and its denominators are named accurately.
- **The anisotropy handling.** The uplands are 0.515 × 0.641 m and `gsd_for(tile_id)` is applied,
  with a self-test that catches its omission. Getting this wrong would have widened every upland band
  by up to 28% and flattered the claim; it is right.
- **D18, no threshold.** Grounds for reinstating a bar were looked for. §11 establishes, and the four
  converted papers confirm, that no published boundary-to-interior rate ratio exists to calibrate
  against. Any bar would be invented. D18 stands.

---

## 6. Forbidden sentences

| Sentence | The design fact that forbids it |
|---|---|
| "Label quality, not model capacity, is the binding constraint" | Architecture is constant in every contrast the design computes, so an architecture-induced boundary limit is unfalsifiable here |
| "Annotation effort is best spent at the class boundaries" *(unconditioned)* | If the boundary residual is mixed pixels at 0.5 m, re-annotation recovers nothing; where a fresh pass would change most labels is not where accuracy can be recovered |
| "The model generalises to unsurveyed terrain" | n = 2 purposive sites, 5.16 km², semi-natural 60% of foreground against 4% in training |
| "Cross-dataset transfer improves X by Y pp" *(as the name of main effect A)* | The transfer arm receives exactly 2.00× the Biodiversity gradient steps |
| "92% / 96% of foreground error falls within 8 m" | Leakage-inflated; and the share form was registered and retracted twice for landscape-dependence (D9 v0/v1) |
| "The class-balanced sampler has no effect" / "is redundant" | D4 — claiming "no effect" needs a bar for what counts as nothing, which D17 forbids |
| "Approximately 88% of masks contain labelling errors" | No recorded protocol; the same figure appears in the project notes as Ortiz et al.'s unrelated inter-annotator bound |
| "Every class collapses to a near-zero interior rate" | Stated at the undeclared 1.5 m / 8 m partition, and the previous campaign's version was flagged internally as unsupportable (D16a) |

---

## 7. What could not be verified

- **Anything about model behaviour.** Nothing has been trained on this split. Every number in
  `main.tex` is withdrawn and was judged only as a set of claims, not as evidence.
- **Cheng et al. 2021 against the CVPR proceedings.** The conversion here is arXiv 2103.16562, as
  `SOURCES_BOUNDARY_LITERATURE.md` records. The two quotations §10 rests on appear verbatim in it.
  Whether the proceedings differ was not checked — that check remains outstanding.
- **Kohli's 320 × 213 image dimensions.** §10 states them; they were not found in the conversion and
  were not confirmed independently. The substantive point (the "8" is pixels, in a caption, offered
  as illustration) is confirmed.
- **Liu et al. 2016** — still paywalled. Still not citable for any number.
- **The ~88% inspection protocol.** No document recording it exists anywhere in `notes/` or `docs/`.
  Both were searched.
- **Whether the campaign actually runs in 12–16 h per task.** `--time=20:00:00` was read from
  `sonic/campaign/campaign.slurm`; nothing has run on this split, so the estimate is the brief's, not
  a measurement.
- **The magnitude of the Test B survey-edge contamination (finding 2).** The mechanism is established
  from the code and the recorded 38% background share. How much it moves rho cannot be known until
  the campaign exists — which is why the foreground-only variant needs to be written now, not after.

---

## Code note

`docs/FIGURES.md` maps `boundary_limited_error.pdf` to `scripts/figures/boundary_limited_error.py`,
whose docstring says *"New-narrative figure; intentionally NOT in build_all_figures.py."* Confirm the
paper's keystone figure actually rebuilds under the documented command.
