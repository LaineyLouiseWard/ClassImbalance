# Decisions log — rebuild, July 2026

Every design decision taken during the rebuild, with the reason and the evidence. Append, do not
rewrite. A decision reversed later stays here with the reversal recorded underneath it.

---

## D1 — Replace the random tile split with a spatially blocked one

**Decision:** spatially blocked, buffered split. **Why:** tiles are chipped on a 50% stride, so ~93%
of each held-out tile's ground was also in training, with byte-identical imagery and labels. Measured
over all 2,143 tiles. Not a judgement call.

## D2 — Two straight cuts, not a block grid

**Decision:** cut the inland site along one axis into train | val | test strips. **Why:** measured, not
asserted — a block grid loses **44.3%** of tiles to buffers against **19.3%** for two cuts.
**Corrected 2026-07-26:** this said 18%. Measured on the shipped manifest, the two cuts drop 413 of
2,143 pool tiles = 19.3%, or 21.2% of the inland site that was actually cut. Quote whichever
denominator the sentence needs, but not 18%. Verified by
running the block-grid path: it returned val=44, test=78.

## D3 — Hold the two upland sites out whole

**Decision:** `external_test`, 191 tiles, never cut. **Why:** a 1,477 m site needs ~1,950 m to pack
train|buffer|test, so neither upland site can carry an internal partition (Roberts et al. 2017). This
is not a fallback — it gives the paper its second estimand, transfer to unsurveyed terrain.

## D4 — Asymmetric buffers: 256 m train|val, 650 m val|test

**Decision:** kept. **Why:** 256 m is the exact pixel-identity distance (512 px at 0.5 m on a 128 m
stride), so beyond it no two tiles share a pixel. Validation only selects checkpoints and is never
reported. **Known weakness:** the final pre-Sonic audit (F6) argues the common-mode defence holds for a
level shift but not for *selection*, since best-epoch choice on a val set 256 m from training rewards
whichever cell best fits the training distribution — and the sampler cells deliberately change that
distribution. Unresolved; must be stated as a limitation.

## D5 — 650 m buffer, below the 950 m measured range

**Decision:** kept. **Why:** the guarantee that reaches the reader is the *realised* separation, 1,664 m
train–test, not the search constraint. Justify with the realised-separation table, not the correlogram.
The pre-registration keys on effect size (peak Mantel r = 0.044), not on the range.
**Outstanding:** the 950 m provenance is not in a fresh clone (`analysis/` is gitignored) and the
committed correlogram artefacts say 750 m and 1350 m. Must be committed and reconciled.

## D6 — Reject splits on independent block support, not class share

**Decision:** `MIN_CLASS_BLOCKS`, with a stricter minimum for test. `MIN_CLASS_SHARE` retired entirely.
**Why:** a share does not track estimability, in either direction. Measured on the folds it produced:
validation semi-natural at 1.91% share had 11 independent blocks and was fine; validation cropland at
7.00% share and 2.6M pixels had 3 blocks and was not. The share floor passes the second and rejects the
first — exactly backwards. Test gets 8 blocks because it is the only split whose per-class numbers are
reported.

## D7 — One split, not three folds

**Decision:** collapsed from three folds to one. **Why:** the folds were role-permutations of one site
and could not be averaged as replicates. Keeping three cost 24 extra runs, training sets differing by
44%, and a paragraph defending bounded test overlap. The decisive measurement: extending the drift term
to validation did NOT fix the val/test composition mismatch, because it is geometric — with three
strips on one axis, one strip contains the cropland cluster and you can only choose which. That defect
only corrupts between-fold comparison, so one fold removes it definitionally rather than repairing it.
**Cost, to be stated:** no empirical handle on sensitivity to cut placement remains.

## D8 — `max_epoch = 45`

**Decision:** 45, not 50. **Why:** `CosineAnnealingWarmRestarts(T_0=15, T_mult=2)` ends cycles at 15,
45 and 105, so 45 stops at an LR minimum. 50 stops mid-cycle and breaks the `save_last` rationale. The
next valid stop is 105. Convergence at 1,072 tiles (6,030 steps against the 9,585 the budget was set
for) is to be confirmed from the first completed campaign run's validation curve, not from a separate
probe.

## D9 — Pre-registration §P1, three versions

**Decision:** rho = (error rate within 8 m) / (rate beyond 8 m), threshold 4.0 on the lower bootstrap
bound. **Why:** two earlier statistics were registered and retracted the same day, both because they
were landscape-dependent.

- v0, error share >= 65%: has a floor equal to the band's area share, 37.8% inland vs 26.5% upland.
- v1, share ÷ area share (which is exactly `lift`): ceiling is 1/area_share, so a common threshold sits
  at different points in each range. It also rises as a model improves, so it partly tests how well the
  model was trained — the alternative hypothesis the claim exists to exclude.
- v2, rho: a ratio of two rates over disjoint pixel sets. No area term, so landscape-independent.

Full history in `docs/PREREGISTRATION_P1_AMENDMENT.md`, dated before any training.
**Outstanding:** the percentile CI under-covers (audit F8); the 4.0 threshold is calibrated on
withdrawn figures and must be re-declared as a priori or its provenance stated.

## D10 — Bootstrap unit is a 950 m spatial block

**Decision:** grid blocks via `utils.spatial_blocks`, shared by all bootstraps. **Why:** two earlier
attempts were wrong. Tile ids treat dependent tiles as independent (294 tiles are 104 disjoint
footprints). Single-linkage over overlapping footprints chains a contiguous strip into ONE component —
294 tiles, 1 unit, degenerate. A grid partitions space instead of chaining through it, and reproduces
the expected counts: 104 units at 256 m, 16 at 950 m.
**Outstanding:** the grid is phase-dependent (audit F2); must be swept and reported.

## D11 — Keep grounded argmax for the OEM→student mapping — 2026-07-26

**Decision:** apply grounded argmax without exception. Bareland → Grassland stands. Do NOT override it
back to Semi-natural.

**Why, in order of weight:**

1. **The audit's decisive reason was measured false.** F1 said two output channels receive "only
   negative evidence for 45 epochs", implying suppression. Measured on the actual stage-2a pool
   (`data/oem_combined_f1/train/masks`, 3,190 tiles): Cropland 21,279,164 px over 248 tiles,
   Semi-natural 11,780,653 px over 261 tiles, both from the Biodiversity half. Both channels receive
   positive evidence throughout. The audit's own figure (Bio half at 7.08% / 4.49%) contradicted its
   interpretation.
2. **What is real is a prior shift, not an absence.** Those classes sit at 0.91% and 0.51% of pool
   foreground against 7.70% and 4.30% in the target training set — about an eighth of target
   prevalence. Correcting a source/target prior shift is what the stage-2b finetune on
   Biodiversity-only exists to do.
3. **The override would reinstate a leak-derived value.** Bareland → Semi-natural was measured at 0.442
   on a training set where 43.3% of the semi-natural pixel mass came from 153 upland tiles now entirely
   in `external_test`. That grounding was partly derived from the generalisation test set.
4. **Grounded argmax is a rule a reviewer can check in one sentence.** An a-priori override sitting on
   the paper's priority class is precisely the thumb on the scale that should be objected to.

**Cost, to be stated in the methods:** OpenEarthMap contributes no Cropland and no Semi-natural labels,
so the transfer arm's effect on those two classes is representation transfer rather than label
transfer. That is falsifiable and, if it holds, stronger than the alternative.

**Note on how the error arose.** A comment I wrote earlier the same day said "the transfer arm
pre-trains with ZERO semi-natural labels", which is false — the pool is Biodiversity + OEM. The
auditor read it, measured the OEM half alone, and built F1's decisive reason on it. Corrected in
`geoseg/taxonomy.py` with the distinction spelled out.

## D12 — The transfer arm is not step-matched; disclose rather than redesign — 2026-07-26

**Decision:** keep the combined Biodiversity + OEM pre-training pool. Do NOT switch to OEM-only
pre-training. Pre-register the step imbalance as a stated limitation and report the gradient-step
counts explicitly.

**The problem (audit F5, verified arithmetic):** the stage-2a pool is 3,190 tiles of which 1,072 *are*
the Biodiversity training tiles. At 1,595 steps/epoch × 45 epochs with a 33.6% Bio share, pre-training
delivers ≈24,100 Bio-tile gradient steps — as many as an entire baseline run. So baseline gets ≈24,120
Bio steps and transfer-only ≈48,240. "OEM transfer" is confounded with a second pass over the training
set. Transfer also receives two val-selection passes to the baseline's one.

**Why disclose rather than fix:**

- The obvious fix — pre-train on OEM only — makes the pool genuinely contain zero Cropland and zero
  Semi-natural, which would create the very channel-suppression problem that D11 established does not
  currently exist. It trades a disclosable confound for a real one.
- It would also invalidate the manuscript's description of the pool and reopen a settled design
  decision at the deadline.
- The paper's contribution is the boundary diagnosis, not the magnitude of the transfer effect. A
  confounded but *declared* transfer effect is survivable; an undeclared one is not.

**What must therefore be done, and is not optional:** register the confound BEFORE results exist, in
`docs/`, stating the two step counts. A pre-registered limitation is defensible; a post-hoc one is not.
If a reviewer demands a step-matched control, the honest answer is that it was identified before
training and declared, with the counts given.

**Reversal condition:** if the transfer effect turns out to be the paper's headline rather than setup,
this is not good enough and a step-matched arm must be run.

## D13 — Test B's NaN and background fractions are survey-edge geometry, not data defects — 2026-07-26

**Decision:** report them as methods content, not as limitations weakening Test B.

**Why:** the auditor recalibrated its own finding. The NaN mask is identical across all four bands and
always touches a tile border, averaging 42% of a Test B tile — that is off-mosaic fill. Tiles are
chipped on a regular grid over an irregular survey footprint, so tiles at the edge of the flown area
are partly empty. Both upland sites are small, so a higher share of their tiles are edge tiles, which
is the whole explanation for Test B's 61.5% labelled area. Nothing is corrupt.

The same effect explains the 9.75% vs 1.70% train/test background gap. Background is the ignore class,
so those pixels never score.

**Supersedes** the framing in `00_STATE.md` and in my earlier reporting, which treated both as
limitations that weakened the generalisation claim. They do not.

**What DOES remain a limitation, narrower than before:** the test strip sits at the far edge of the
site, so it is not a random sample of the surveyed landscape. State that in §2 with the buffer-drop
comparison.

## D14 — My own tiering of the final audit, where it differs from the auditor's — 2026-07-26

The auditor recalibrated its report into three tiers. I checked each call rather than accepting it.
Three of its judgements do not survive, and one of mine did not either.

**Agreed, with a better reason than the one given.** The NaN downgrade is right, but not because the
pixels are off-mosaic fill rather than corruption. Measured: no-data pixels carry a foreground label in
0.00% of train, 0.00% of test and 0.13% of external_test pixels. They are labelled background, and
background is the ignore class, so they never enter the loss or any metric. The residual effect is
contextual only — the encoder sees dark regions through its receptive field in 80% of Test B tiles
against 8% of training tiles. One sentence in the methods. Same conclusion for the train/test background
asymmetry.

**Disagreed 1 — the class mapping is not "the only serious finding".** Its stated mechanism was measured
false (D11). The auditor's restatement, that "the transfer factor isn't what the paper says it is", is a
disclosure obligation about what OEM contributes, not a reason to rebuild the pool. No pool rebuild
follows, so no run is wasted. There is therefore no tier-1 *scientific* blocker at all — only plumbing
and two documentation acts.

**Disagreed 2 — the step confound was dropped from every tier, and should not have been.** Verified from
the imported configs, not from the audit: baseline 536 steps/epoch × 45 = 24,120 Biodiversity gradient
steps; stage 2a 1,595 steps/epoch × 45 × (1,072/3,190 Bio share) = 24,120 more. Transfer arm 48,240
against baseline 24,120, a ratio of exactly **2.00×**, and exact by construction since the Bio tiles are
seen once per epoch either way. This is a confound on main effect A, the paper's headline factor. It is
neither plumbing nor a write-up decision. Per D12 it is handled by registration rather than redesign —
but registration must happen BEFORE results exist, which makes it tier 1 by timing even though it
changes no code.

**Disagreed 3 — 950 m is not just an analysis choice to revisit after the runs.** It is also the block
size in `MIN_CLASS_BLOCKS`, the criterion that admitted this split. Measured on the shipped split:

    650 m   train 33  val 12  test  9   PASSES
    750 m   train 29  val 10  test  9   PASSES
    950 m   train 23  val  6  test  8   PASSES
    1350 m  train 14  val  8  test  6   FAILS

So the split's admissibility depends on which correlogram number is used, and it fails under the
spectral one. **Resolution, which is reasoning rather than re-cutting:** block support is a
class-composition criterion, so the composition range is the right denominator. **Corrected 2026-07-26:**
an earlier version of this paragraph said "750 m on the shipped subsample, 950 m on the full pool".
There is no full-pool inland measurement. The committed correlogram gives the inland site 750 m for
composition and 1,350 m for spectral similarity; 950 m is ireland2's composition range. 950 m sits
ABOVE the inland composition range, so it counts fewer independent units than that scale would and
is conservative for this criterion. The split passes at 650, 750 and 950 m. See METHODS §4. The 1350 m figure is the *spectral*
range and answers a different question (imagery similarity, not class-composition independence). This
argument must be written into the methods now. No re-cut is needed, but it cannot be left to be
discovered under review.

**My tier 1, therefore:** the factorial wiring, the A1b/A2 ordering, registering the step confound, and
writing the block-size justification. Two code bugs and two documentation acts.

## D15 — rho threshold stays at 4.0, justified from our own pilot estimates — 2026-07-26

**Decision:** keep rho >= 4.0. Justify it from this project's own preliminary estimates. Do NOT search
the literature for a reference value; do not commission further work on this.

**Why.** A threshold cannot be set from the campaign it will judge — that is circular and is what a
pre-registration exists to prevent. But it can be set from other data, and we have some: rho measured on
held-out ground from an earlier run, on pixels no training tile covered, giving 3.25 (baseline model),
4.77 (full model, validation) and 12.02 (full model, test). Setting the bar at 4.0 puts it below the
weakest full-model estimate and above the baseline's, so the baseline fails and the full model is at
genuine risk. That is what makes the registration binding rather than decorative, and it is the only
property the threshold really has to have.

Calibrating a pre-registered threshold on pilot data is standard practice. I briefly proposed searching
the literature for an externally-derived value instead, and withdrew it: no such benchmark is known to
exist for this quantity, reaching for new citations at this stage signals unsettled scope when the scope
is in fact settled, and it would have cost time against a deadline for a wording problem rather than a
methodological one.

**The methods sentence:** "We pre-registered a threshold of rho >= 4 from preliminary estimates on
held-out ground, which ranged from 3.3 to 12.0. The bar sits below the weakest estimate for the full
model and above that for the baseline, so the baseline is expected to fail it."

**What is NOT claimed:** that 4.0 is derived from theory or from published benchmarks. It is a pilot-
calibrated bar, stated as such.

## D16 — Drop the pre-registration. Report the curve. — 2026-07-26

**Decision:** no pre-registered threshold. The boundary claim is supported by the evidence the paper
already computes — the trimap exclusion curve and the boundary-versus-interior error rates — reported
descriptively. `docs/PREREGISTRATION_P1_AMENDMENT.md` is withdrawn.

**Why.**

A pre-registration is not standard in remote sensing or computer vision; it is standard in clinical
trials and psychology. It was adopted here as credibility armour after the first campaign was withdrawn
for leakage, not because the field expects it. That made it optional, and it was not worth its cost.

The cost was compounding. rho required a threshold. The threshold required a provenance. The provenance
was awkward because the only calibration data came from the withdrawn campaign, so it needed either a
literature search or a further justification — and every layer added a number that could be argued
with, on top of a plot that already showed the effect directly. Three layers of machinery over evidence
that was already legible.

The plot is also the better evidence. The trimap curve shows accuracy as a function of how much of the
boundary band is excluded; the reader sees the whole shape rather than one number and one arbitrary
bar. It is what Kohli, Ladicky & Torr (2009) and Csurka, Larlus & Perronnin (2013) established for
exactly this purpose, it is already implemented in `boundary_trimap_iou.py`, and it is already cited.

**What is kept.**

- The trimap exclusion curve, as the primary evidence.
- Boundary-band and deep-interior error rates per class, reported as rates, so no area denominator and
  no landscape-dependence enters.
- rho may still be quoted as a one-line summary of that table. It is descriptive. There is no threshold
  and nothing "fails".
- Per-class support (pixels, tiles, independent blocks) beside every per-class number, with thin classes
  marked unestimable. That was never part of the pre-registration and stands on its own.

**What is dropped.**

- The rho threshold of 4.0, the 2.0 dead band, and the weak band.
- The registered second arm. The cross-cell comparison of boundary and interior rates is still worth
  reporting, as an observation.
- Items 15, 16 and 17 of the implementation brief, and its `do_this_first` interval-coverage check.

**Legitimacy of withdrawing it.** `docs/PREREGISTRATION_P1_AMENDMENT.md` is committed and timestamped,
so this withdrawal must be explicit rather than quiet. It is withdrawn **before any model has been
trained on the corrected split**, so no result has been seen and nothing is being avoided. Withdrawing
after seeing results would not be legitimate; withdrawing now is simply deciding not to use an optional
instrument.

**The honest reading of how this arose.** I introduced the threshold, then spent a working session
solving problems the threshold created — its landscape-dependence, twice, and then its provenance. The
underlying evidence never needed it. This is the clearest instance in the rebuild of machinery accreting
around a decision instead of the decision being questioned.

## D17 — Stop inventing numeric thresholds — 2026-07-26

**The pattern, which is the most repeated mistake of this rebuild.** A number gets reported, then a
threshold gets invented to classify it, then the threshold needs defending, then the defence needs its
own machinery. Instances, all mine:

- `rho >= 4.0`, plus a 2.0 dead band and a weak band. Withdrawn (D16) after two rounds of solving
  problems the threshold itself created.
- `report_class_support.py`'s ok / weak / UNESTIMABLE labels, against `MIN_BLOCKS = 5`,
  `MIN_TILES = 8` and a "weak" band at twice those. Three invented numbers.
- `min_val_tiles = 100`, `min_test_tiles = 150`.
- `MIN_CLASS_BLOCKS = 5` and `min_test_class_blocks = 8`.
- `MIN_CLASS_SHARE`, `max_test_overlap = 0.25`. Both already retired.

**The rule from here:** report the number; do not invent a bar for it. A threshold is justified only
when it must *decide* something automatically. A gate that stops a bad split reaching training is a
legitimate threshold. A label on a results table is not — it replaces the reader's judgement with a
number I chose, and then has to be defended as though it were derived.

**Applied now:** the verdict labels are removed from `report_class_support.py`. It reports share,
pixels, tiles and independent blocks, and stops. Test B cropland is 30 tiles in 4 blocks; that is the
finding, and it speaks without being called UNESTIMABLE.

**Not reversible:** `MIN_CLASS_BLOCKS` and `min_test_class_blocks` already selected the shipped split.
They stay, but they are described in the methods as a design constraint — each class had to appear in
at least N independent locations — and not as a derived criterion.

**Footnote on how easily this goes wrong.** Removing the labels left two references to the deleted
constants. `py_compile` passed, because Python does not resolve names at compile time, and I reported
"compile ok" on a file that would have raised at runtime. Caught by actually running it. Compiling is
not running.

## D16a — The withdrawal in D16 is REOPENED, not settled — 2026-07-26

D16 withdrew the pre-registration on my own judgement. That was a decision to hand over, not to take:
it changes what the paper claims and how, and it was made in one exchange.

`docs/BRIEF_RHO_THRESHOLD_PROVENANCE.md` puts it to an independent chat with both sides argued —
including the case AGAINST withdrawing, which nobody had made properly: that a timestamped commitment
is the only clean answer to the suspicion that a second set of numbers was tuned after a first set was
withdrawn for leakage, and that without a bar the claim is settled by interpretation, which is what
produced the unsupportable "every class collapses to a near-zero interior rate" in the first place.

Until that comes back, treat the pre-registration as **suspended, not withdrawn**.
`docs/PREREGISTRATION_P1_AMENDMENT.md` keeps its withdrawal header and its full history; if the
decision reverses, the header is removed and the reversal is dated in the open.

Nothing has been trained, so either outcome remains legitimate.

## D18 — No threshold. Settled by the author, not by the brief — 2026-07-26

**Decision:** no pre-registered threshold. D16 stands; D16a's reopening is closed without running
`docs/BRIEF_RHO_THRESHOLD_PROVENANCE.md`. That brief is superseded, not pending.

**Why, in the author's words:** the number is arbitrary. A bar nobody can source is distracting and
overpromises what it can adjudicate.

**What the measurement adds, having been done anyway:** a bar of 4.0 judged on a lower bound is not a
bar of 4.0 — its operating point is ~5.2 on Test A and ~6.3 on Test B at 16 and 12 blocks. Keeping it
would have required registering that operating characteristic and replacing the percentile interval
with a block jackknife, since the percentile under-covers and BCa is worse rather than wider. Three
pieces of apparatus to defend one arbitrary number. The measurement supports the decision; it did not
drive it.

**What carries the claim instead:** the trimap exclusion curve, boundary-band versus deep-interior
error rates per class, and per-class support beside every number. rho may be quoted descriptively.
Nothing "fails".

**Consequence:** the interval-coverage work is no longer decision-critical and its simulation stays in
session scratch rather than being productionised. Had the threshold been kept it would have needed to
be in the repository and on Zenodo.

## D19 — Statistical reproducibility, not bitwise. Decided, not deferred — 2026-07-26

**Decision:** do not pursue bitwise seed reproducibility. Do not set
`torch.use_deterministic_algorithms(True)` or `Trainer(deterministic=True)`, and keep
`precision="bf16-mixed"`.

**Why.** Bitwise determinism is unattainable here regardless: bf16 mixed precision makes reduction
order hardware-dependent, so the same seed on a different card gives different bits, and the campaign
runs on a shared cluster. Chasing it would mean pinning hardware, losing speed, and hitting ops with
no deterministic kernel — apparatus bought for a property no reader asks for. The same test that
retired the rho threshold (D18) applies: machinery with nothing standing behind it.

What the paper claims is a distribution over ten seeds with its spread reported, which is the
reproducibility standard that actually applies. `cudnn.deterministic=True`, `cudnn.benchmark=False`,
`seed_everything`, `pl.seed_everything(workers=True)`, seeded `worker_init_fn` and generators, and
seeding before `py2cfg` are all already in place and were verified to work.

**What was done instead, because both are cheap and both have standing:**

1. **Run provenance.** See item 2. (An earlier version of this entry claimed the seeds do not vary
   initialisation, on a measurement of all 432 parameter tensors being identical. **That claim was
   withdrawn on 2026-07-26**: it was measured inside a single Python process, and `py2cfg` caches the
   module, so the second build reused the first network. Re-measured in separate processes, 46 of 432
   tensors differ in the baseline and sampler-only cells and 171 in the two transfer cells. The seeds
   are genuine independent draws of the student pipeline, as originally documented. Corrected in the
   methods list, §9. The decision above is unaffected — it never depended on this.)
2. **Run provenance.** Nothing anywhere recorded what produced a checkpoint. Each run now writes
   `run_provenance_seed<N>.json` beside its weights: commit and dirty flag, GPU, precision, torch and
   lightning versions, cuDNN flags, split tag and seed. A number is attributable only if the
   conditions travel with it.

**Also settled here:** convergence at 1,072 tiles is assumed rather than probed. The 45-epoch budget
ends at an LR minimum (D8) and a separate convergence study is not worth the GPU time against the
deadline. If the first run's validation curve is still climbing it will be visible in the logs that
are kept anyway.
