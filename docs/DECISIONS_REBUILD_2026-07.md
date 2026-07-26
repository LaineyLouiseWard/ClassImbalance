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
asserted — a block grid loses **44.3%** of tiles to buffers against **18%** for two cuts. Verified by
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
