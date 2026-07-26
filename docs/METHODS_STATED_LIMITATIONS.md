# Things the methods section must state, with the measurement behind each

Started 2026-07-26, **before any model was trained on the corrected split**. This is the working list
of properties of the design that a reader is entitled to know and that the results cannot be read
correctly without. Each entry gives the number and how it was obtained, so §2 can be written from
here rather than from memory.

Two rules for this file. Every number is recomputed from the shipped configs or from the tiles on
disk — never from a code comment, a note, or the withdrawn leaking campaign. And an entry is added
when the property is discovered, not when the results make it convenient.

Related records: `docs/DECISIONS_REBUILD_2026-07.md` (why each design decision was taken),
`docs/PREREGISTRATION_P1_AMENDMENT.md` (SUSPENDED 2026-07-26 pending D16a, retained in full).

---

## 1. The transfer arm receives exactly twice the Biodiversity gradient steps

**Measured 2026-07-26, from the imported configs and the tiles on disk.**

Stage 2a pre-trains on a combined pool. Counted on `data/oem_combined_f1/train/images`: the pool is
3,190 tiles, of which **1,072 are the Biodiversity training tiles themselves** — pool ∩ train = 1,072,
train − pool = 0. Pre-training is therefore not a pass over a foreign dataset. It is a pass over
OpenEarthMap *and* a second pass over the training set.

| cell | tiles/epoch | steps/epoch (batch 2) | epochs | Biodiversity gradient steps |
|---|---|---|---|---|
| baseline (`stage1_baseline`) | 1,072 | 536 | 45 | 24,120 |
| pre-train (`stage2a_oem_pretrain`) | 3,190 | 1,595 | 45 | 24,120 — the 33.605% Bio share of 71,775 |
| finetune (`stage2b_oem_finetune`) | 1,072 | 536 | 45 | 24,120 |

**Baseline 24,120. Transfer arm 24,120 + 24,120 = 48,240, a ratio of exactly 2.00×.** Exact rather
than approximate: every Biodiversity training tile is seen once per epoch in both stage 2a and stage
2b, so the transfer arm passes over the training set 90 times against the baseline's 45. The same
holds on the other level of the sampler factor — `stage_sampler_only` 24,120 against `stage3_clsbal`
48,240 — so the confound sits squarely on main effect A.

**What must be written:** main effect A is not "cross-dataset transfer". It is *transfer plus a
second pass over the training set*, and this design cannot separate the two. A reader may not
attribute the transfer contrast to cross-dataset representation alone.

**Why it was not fixed rather than declared** (D12): the step-matched control would pre-train on
OpenEarthMap only, which removes the Biodiversity half of the pool and with it every Cropland and
Semi-natural label — see §3 below. That trades a declared confound for an undeclared one. Identified
and written down before any results existed, which is the only thing that makes a limitation
defensible rather than an excuse.

## 2. The transfer arm receives two checkpoint-selection passes to the baseline's one

**Measured 2026-07-26 by importing all five configs.** Every one monitors `val_mIoU` on the same
173-tile validation split with `save_top_k=1`. Stage 2a selects a best epoch on that split; stage 2b
then selects again from the model 2a handed over. Baseline and sampler-only select once.

Validation sits 256 m from training, and its per-class support is thin: 8, 8, 6, 7 and 6 independent
950 m blocks for Forest, Grassland, Cropland, Settlement and Semi-natural. Selection is on foreground
mIoU, which weights all five equally, so it is partly driven by a Cropland IoU estimated from 6
blocks. The usual defence — that validation optimism is common-mode across the cells — holds for a
level shift but not for a *selection rule*, and the arm that selects twice is the arm carrying the
positive result.

**What must be written:** the selection asymmetry, alongside §1. It is declared, not corrected.

## 3. OpenEarthMap contributes no Cropland and no Semi-natural labels

**Measured 2026-07-26 over all 3,190 pool masks.** Foreground shares within each half of the pool:

| | Forest | Grassland | Cropland | Settlement | Semi-natural |
|---|---|---|---|---|---|
| OpenEarthMap half (2,118 tiles) | 27.044% | 54.413% | **0.000%** | 18.542% | **0.000%** |
| Biodiversity half (1,072 tiles) | 14.579% | 70.111% | 7.703% | 3.342% | 4.265% |
| pool overall | 25.566% | 56.275% | 0.914% | 16.739% | 0.506% |

The grounded argmax maps Bareland, Rangeland and Agriculture all to Grassland, so no OpenEarthMap
class lands on Cropland or Semi-natural.

**What must be written:** for those two classes the transfer factor is *representation* transfer, not
label transfer.

**What must NOT be written:** that the two output channels are suppressed. They are not. Both receive
positive evidence throughout stage 2a from the Biodiversity half — Cropland 21,279,164 px across 248
tiles, Semi-natural 11,780,653 px across 261 tiles. What is real is a prior shift: 0.914% and 0.506%
of pool foreground against 7.703% and 4.265% in the target training set, about an eighth of target
prevalence. Correcting a source/target prior shift is what the stage-2b Biodiversity-only finetune
exists to do.

## 4. The 950 m block size, where it comes from, and what the split's admissibility depends on

**Measured 2026-07-26.** `SUPPORT_BLOCK_M = 950.0` is not only the bootstrap unit. It is also the
block size in `MIN_CLASS_BLOCKS`, the criterion that ADMITTED this split, so the split's
admissibility is a function of it.

Recomputed against the shipped manifest by `scripts/analysis/block_size_sensitivity.py`. Each cell
is the smallest number of independent blocks any one foreground class has in that split; the floors
are 5 for train and val and 8 for test.

| block | train | val | test | verdict | where the number comes from |
|---|---|---|---|---|---|
| 650 m | 33 | 12 | 9 | passes | ireland1 composition range; also the val/test buffer width |
| 750 m | 29 | 10 | 9 | passes | **inland composition range**, 900 of 1,952 tiles |
| 950 m | 23 | 6 | 8 | passes | shipped value |
| 1350 m | 14 | 8 | 6 | **fails** | **inland spectral range** |

The 950 m row reproduces the manifest's own declared `class_block_support` exactly, class by class
and split by split.

**Where 950 m actually comes from, stated plainly.** It is not the inland site's measured
correlogram range, and three code comments said it was until they were corrected on 2026-07-26.
The committed measurements (`artifacts/correlogram/`, Mantel correlogram, 100 m increments, 9,999
permutations) give, for the inland site, a composition range of **750 m** and a spectral range of
**1,350 m**, both on a 900-of-1,952-tile subsample. **950 m is ireland2's composition range** — one
of the two upland sites. No artefact reports 950 m for the inland site.

**Why the split is nonetheless admissible, and why this is reasoning rather than re-cutting.** Block
support is a criterion about *class composition*: it asks how many independent parcels of ground
carry a class, so that a per-class number is not one place measured repeatedly. The scale that
applies to it is therefore the composition range, which is 750 m inland. 950 m is above that, so it
counts *fewer* independent units than the criterion's own scale would and cannot flatter the
support. The split clears the floors at 650, 750 and 950 m.

At 1,350 m it fails. That is the **spectral** range, and it answers a different question — how far
apart two tiles must be before their imagery stops looking alike, not before their class composition
becomes independent evidence. A split admitted on class composition is not obliged to clear a bar
set by imagery similarity, and the two ranges differ by 1.8× on the same site.

**What must be written:** the table above, the provenance of 950 m including that it is another
site's number, and the composition-versus-spectral distinction. A reader who finds the 1,350 m
failure themselves and finds no mention of it in the paper will reasonably conclude it was hidden.

**What must NOT be written:** that 950 m is the inland site's measured range, or that it is a
full-pool measurement of which 750 m is the subsample. There is no full-pool inland measurement.

## 5. The 950 m block grid is phase-dependent, and the split's own gate turns on the offset

**Measured 2026-07-26 by `scripts/analysis/block_phase_sweep.py`.** The grid is anchored at the CRS
origin — for the inland site, the UTM 29N false easting of 500,000 m. That origin has nothing to do
with the landscape, so the shipped partition is one member of a family indexed by the offset.

Ten offsets, 0.1 of a cell (95 m) apart, both axes together:

| split | shipped | min | max |
|---|---|---|---|
| train | 32 | 28 | 40 |
| val | 8 | 7 | 16 |
| test (Test A) | 16 | 8 | 16 |
| external_test (Test B) | 14 | 10 | 14 |

Two things rest on that offset. Interval width scales roughly as 1/sqrt(n_blocks), so the same data
at an equally arbitrary offset gives a Test A interval up to **1.41× wider**. And the adequacy
criterion moves with it: **the shipped split clears its own class-support floors at 5 of the 10
phases.** The shipped phase sits at the top of the Test A range.

**What must be written:** the sweep, as a sensitivity. The split is not wrong — no offset is more
correct than another — but a criterion that passes at half the offsets of an arbitrary grid is a
property of the criterion, not of the ground, and belongs in the methods rather than in a reviewer's
discovery.

## 6. Nominal block counts overstate the support; report Kish n_eff beside them

**Measured 2026-07-26, same script.** The blocks are badly unequal, so resampling them as
exchangeable draws claims more independence than there is.

| split | tiles | blocks | n_eff (tiles) | n_eff (foreground px) | tiles per block |
|---|---|---|---|---|---|
| train | 1072 | 32 | 27.46 | 26.93 | 55 … 8 |
| val | 173 | 8 | 7.52 | 7.43 | 28 … 12 |
| test (Test A) | 294 | 16 | **9.85** | 9.77 | 43, 36, 36, 35, 34, 33, 18, 17, 8, 7, 7, 6, 5, 4, 3, 2 |
| external_test (Test B) | 191 | 14 | **7.15** | 5.78 | 42, 37, 34, 15, 14, 14, 9, 6, 6, 4, 4, 2, 2, 2 |

Restricted to the 172 Test B tiles that carry a ground-truth boundary, the count is 12 blocks and
n_eff **7.36** by tiles — independently reproduced twice from the raw rasters.

**What must be written:** n_eff beside n_blocks wherever a block bootstrap is reported. Six of Test
A's sixteen blocks hold 74% of its tiles.

## 7. Test A's interval is not what a nominal 95% interval claims

**Measured 2026-07-26 by simulation on the real per-block band and interior pixel counts**, and
independently re-derived from the rasters by a second implementation sharing no code. Error is drawn
per block with a log-normal random effect on both the interior rate and the rate ratio.

- The **percentile** interval under-covers at these block counts: 0.86–0.93 against a nominal 0.95,
  and 0.92–0.93 even with zero between-block heterogeneity, which is six Monte-Carlo standard errors
  low. The miss is asymmetric and on the side that matters — the interval sits entirely above the
  truth about 7% of the time on Test A against a nominal 2.5%.
- **BCa does not fix it and is worse**, which contradicts the assumption that a better interval would
  be wider. BCa's median width is 1.012× the percentile's, its *lower* bound is HIGHER in 63 of 64
  simulated cells, and its coverage falls to 0.74–0.86 — with 12 blocks the acceleration is estimated
  from 12 jackknife points and is unstable.
- A **delete-one-block jackknife t-interval on log** is the only one of the three that covers near
  nominal (0.97 with no heterogeneity, 0.89–0.94 with it), and it is 20–45% wider.
- The claim that a percentile interval on log(rho) is *bit-identical* to one on rho is false, though
  only at 5e-8 — quantile interpolation on a convex transform does not commute. It is numerically
  irrelevant and the word should not be written down.

**Why this matters beyond the interval.** A threshold judged on a lower bound is not the threshold it
appears to be. For a lower bound to clear 4.0 with 80% probability, the true rate ratio must be about
**5.2 on Test A and 6.3 on Test B** under moderate heterogeneity, rising to 5.8 and 7.0 under strong
heterogeneity, and 5.5 / 7.5 if the properly-covering jackknife interval is used. Test B needs roughly
1.0–1.4 more than Test A for the same power, purely because it has fewer and more unequal blocks. The
only non-leaking prior estimates available were 3.25 (baseline) and 4.77 (full model) on validation.

**What must be written:** if a threshold is used at all, its operating characteristic, computed
before the campaign. If one is not, the coverage of whatever interval is reported, and n_eff.

## 8. The study is three Irish sites; the raw pool holds eleven

**Measured 2026-07-26 from the GeoTIFF geometry, not from a filename.** `data/biodiversity_raw`
contains 2,307 image/mask pairs across eleven site prefixes. The study uses three:

| site | tiles | centroid | |
|---|---|---|---|
| `biodiversity` | 1,952 | 52.60 N, 8.65 W | inland Ireland, UTM 29N, 0.500 × 0.500 m |
| `ireland1` | 64 | 51.55 N, 9.63 W | upland Ireland, WGS84, anisotropic |
| `ireland2` | 127 | 52.04 N, 9.26 W | upland Ireland, WGS84, anisotropic |
| `col1` | 36 | **4.77 N, 74.23 W** | **Colombia — excluded** |
| `den0`–`den6` | 128 | **55.10 N, 8.80 E** | **Denmark — excluded** |

Excluding the 164 Colombian and Danish tiles is correct and was decided when the pool was built: a
different biome, different field structure and different acquisition have no place in a claim about
Irish rural land cover. **The exclusion was not the problem. Nothing enforcing it was.**

It had been applied by hand to the built pool and left no record in code, so the stage that unpacks
the raw tiles passed the whole directory. A from-scratch run therefore rebuilt a 2,307-tile pool,
not the 2,143-tile pool every pool-level number in this repository was measured on — including the
correlogram's 1,952 inland tiles and the "2,143 tiles" the leakage measurement was made over.

The site list is now named in `split_biodiversity_dataset.py` (`STUDY_SITES`), and the exclusion is
applied after the shuffle and the slicing, which is where it was applied historically. A from-scratch
run now reproduces the shipped pool exactly: 1706 / 219 / 218, the same 2,143 ids, **zero tiles
changing directory**. Applying it any earlier moves 543 tiles between pool directories and strands
435 of `data/split_f1`'s 1,730 symlinks, because those symlinks resolve through the pool's directory
layout even though the assignment itself is discarded.

**What must be written:** the study is three sites of eleven in the delivered data, and which two
were excluded and why.

## 9. What the ten seeds actually vary

**Corrected 2026-07-26. An earlier version of this section claimed all 432 parameter tensors were
identical across seeds. That was wrong**: it was measured by building two configurations inside one
Python process, and `py2cfg` caches the imported module, so the second build reused the first
network object. Re-measured in separate processes:

| cell | tensors differing between seed 42 and 43 | why |
|---|---|---|
| `stage1_baseline` | **46 of 432** | decoder head randomly initialised; Swin backbone loaded from `stseg_base.pth` |
| `stage_sampler_only` | **46 of 432** | same |
| `stage2b_oem_finetune` | **171 of 432** | constructed with `pretrained=False`, then warm-started from stage 2a |
| `stage3_clsbal` | **171 of 432** | same |

So the seeds are genuine independent draws of the student pipeline: the randomly-initialised decoder
head, the sampler draw order and the augmentation RNG all vary. A repeated seed reproduces on the
same machine.

**What must be written:** the ten-seed spread covers decoder initialisation, sampler order and
augmentation, at a fixed pretrained backbone. The backbone is held constant by design across all
four cells, which is what makes the paired contrasts interpretable.

**Bitwise reproducibility is not claimed, and is not pursued.** `precision="bf16-mixed"` makes
reduction order hardware-dependent, so identical numbers across machines are unattainable without
pinning the GPU, which a shared cluster cannot offer. Forcing
`torch.use_deterministic_algorithms(True)` would buy a property no reader needs at the cost of
training speed and of ops that have no deterministic kernel. What is claimed instead is statistical
reproducibility — the seed set, the spread over it, and the conditions that produced it. Each run
writes `run_provenance_seed<N>.json` beside its checkpoint: commit, whether the tree was dirty, GPU,
precision, torch and lightning versions, cuDNN flags and the seed.

**The lesson, which is the same one this repository keeps teaching:** the false version was produced
by a check run in the same process as the thing it checked. Building both models in one interpreter
could not have detected a reseeding failure, because the module cache guaranteed the answer.
