# Seventh audit — findings, 2026-07-26

Run before the GPU campaign, against `8c54599`. Method: every dimension-1 quantity re-derived from
the rasters and configs with standalone code importing nothing from `scripts/` or `geoseg/`, written
down before any document was read; every claimed defect demonstrated by running a command, not by
reading one. No tracked file was modified; `git status --porcelain` empty at exit.

Caveat on independence: `CLAUDE.md` is auto-injected as project instructions, and the audit brief
itself quotes figures, so the split sizes, buffers, 950 m, the band shares, "19 of 191" and the
"12 and 14" blocks had all been seen before deriving. Those rows are verification. The rows derived
with no prior exposure are the pool composition, the stage-2a pool, per-class support, Kish n_eff,
the phase sweep, the gradient-step ratio, the boundary-free identities and the OEM class composition.

---

## Verdict

**The campaign can launch.** Training is correctly gated. Fix item 1 first — it is ten minutes now
and expensive after.

What was attacked and held: the leakage gate rejects a known-bad input (a train tile whose raster is
actually test ground, with every id, count and pairing intact — caught by both the separation and the
overlap check, exit 1); the stage reorder is real in file order, not just in the `STAGES` array; the
provenance block executes against all five configs; `SPLIT_TAG` is exported by both launchers; the
taxonomy guard passes 31 checks; `Evaluator.Intersection_over_Union` and `metrics_from_cm` agree on
every edge case including absent classes; `evaluation_results/val` and `/test` are genuinely empty.

Dimension 1 reconciled **to the pixel** on every quantity except two, and both have one cause (§6).
Notably Test A's registered band denominator matched to fifteen significant figures.

---

## Fix list

Ordered by when it has to happen, not by how interesting it is. Everything found is here; none of it
is filed as too small to bother with.

**Before the array is submitted**

| # | fix | where |
|---|---|---|
| 1 | make C5's `--out-dir` and stage D's `--softmax-root` resolve to the same place | `RUNBOOK.sh`, `seed_disagreement.seed_dir` |
| 2 | collect `analysis/seed_softmax/**` off each seed, and write a fetch script | `sonic/campaign/campaign.slurm`, `sonic/campaign/` |
| 3 | exit non-zero when `n_tiles == 0` | `boundary_trimap_iou.py` |
| 4 | add the A0 taxonomy check to the cluster preflight | `campaign.slurm` |

**Before submission — code**

| # | fix | where |
|---|---|---|
| 5 | provenance guards must also assert `split` and `data_root` | `export_final_test_table.py`, `aggregate_seeds.py:265`, `campaign.slurm:117` |
| 6 | derive per-site GSD once; delete the four duplicated `GSD_BY_SITE` tables | `seed_disagreement.py`, `boundary_exposure.py`, `figure_label_ceiling.py`, `figures/boundary_limited_error.py` |
| 7 | `gsd_for()` must raise on an unknown site, not fall back to `(0.5, 0.5)` | `seed_disagreement.py:77` |
| 8 | one module for `111320`/`111132`; fix the latitude constant | 7 files; bug at `terrain_separability.py:75` |
| 9 | remove the hardcoded read of the withdrawn `analysis/eval_219/` | `boundary_exposure.py:74` |
| 10 | make `BIO_OEM_COMBINED` required, like `BIO_SPLIT` | all five configs |
| 11 | make `--data-root` required (default is the withdrawn split) | `compute_metrics.py:311` |
| 12 | replace the bare `except Exception: return None` | `boundary_rate_ratio.py:289` |
| 13 | make the 950 m grid anchor an explicit named constant | `utils.py`, `build_spatial_split.py` |
| 14 | don't print `ok` beside a check that failed | `assert_no_split_leakage.py:348` |

**Before submission — text and numbers**

| # | fix | where |
|---|---|---|
| 15 | report Kish n_eff, not 14, as Test B's unit count; add the 5.23 km² / 5.79-block area equivalent | METHODS §6, §7, manuscript |
| 16 | state that `MIN_CLASS_BLOCKS` does not cover external_test, and that Test B Cropland is 4 blocks | METHODS §4 |
| 17 | drop the stale "reported as unestimable" claim (removed under D17) | `RUNBOOK.sh:442` |
| 18 | "4 of 12" → 4 of 14 | `bootstrap_metrics.py:114`, `geoseg/utils/metric.py:98` |
| 19 | stage 2a is 1595 steps/epoch, not 536 | `stage3_clsbal.py:166` |
| 20 | document the train split's boundary-free tile `biodiversity_0808` | METHODS, CLAUDE.md |
| 21 | open item 1 is resolved; the "1,846 tiles" note is stale | CLAUDE.md |
| 22 | the realised val–test separation is 768 m, not the requested 650 m | CLAUDE.md, METHODS |
| 23 | cut the "100.0000% identical" sentence — it is a tautology of the chipping | METHODS §10 |
| 24 | ledger rows C5 and C7 are marked DONE and are not | `PRE_SUBMISSION_LEDGER.md` |

---

## 1. LAUNCH-BLOCKING — the boundary evidence has no working path

Stage C5 writes softmax to `analysis/seed_softmax/<CELL>/seed<N>/`.
`seed_disagreement.seed_dir()` reads `<root>/seed<N>/analysis/seed_softmax/<CELL>/seed<N>/`.
Stage D passes `--softmax-root analysis/seed_softmax`, which resolves to
`analysis/seed_softmax/seed<N>/analysis/seed_softmax/<CELL>/seed<N>/` — nothing.

Run with the real paths, `boundary_trimap_iou.py` — the primary evidence for the boundary claim
since D18 — produced:

```
n_tiles: 0,  n_seeds: 10,  every IoU NaN
wrote boundary_trimap_stage3_clsbal.json + boundary_trimap_preview.png
EXIT=0
```

It writes results and a figure, asserts a ten-seed ensemble it never loaded, and does not trip
`set -euo pipefail`. Its sibling `boundary_rate_ratio.py` exits 1 on the same input. Same failure,
one shouts and one whispers, and the one that whispers is the primary evidence.

Compounding it: `sonic/campaign/campaign.slurm` collects only `metrics.json`, and `run_campaign.sh`
symlinks only `data` and `pretrain_weights` into each seed worktree. Every seed's dumps stay stranded
in its own tree, and `sonic/campaign/` has no fetch script at all. If scratch is purged the dumps are
gone and recovering them means re-running inference over 40 checkpoints.

**Fix:** make the C5 `--out-dir` and the stage D `--softmax-root` meet; collect
`analysis/seed_softmax/**` in campaign.slurm's collect step; make
`boundary_trimap_iou.py` exit non-zero when `n_tiles == 0`. This ledger row (C7) is currently marked
DONE.

## 2. Provenance guards validate the model but never the data

Took the withdrawn campaign's own metrics, changed **only the checkpoint basename**, placed them at
the current tagged path, ran `export_final_test_table.py`:

```
mIoU = 80.2%   EXIT=0   final_test_table.tex written
```

The accepted files' own fields said `split: val`, `data_root: .../biodiversity_split/val` (the
withdrawn leaking split), `date: 2026-06-19`. One of the two cells was the retired `stage3_sampler`
relabelled as `stage3_clsbal`. `aggregate_seeds.py:265-274` carries a byte-identical guard with the
same hole — duplicated rather than shared, so one blind spot exists twice. `campaign.slurm`'s
per-seed check (line 117) has it too.

**Fix:** in all three, additionally assert `m["split"]` is the expected split and that
`m["data_root"]` resolves under the current `$SPLIT_ROOT`. Both fields are already in the file.
Ledger row C5 is marked DONE; it is half-done.

## 3. Hardcoded physical constants, in seven files, slightly wrong

This is structural, not cosmetic. The same derivable quantity is hand-typed in many places, and the
hand-typed values are wrong.

`GSD_BY_SITE = {"ireland1": (0.515, 0.641), "ireland2": (0.515, 0.634)}` is duplicated verbatim in
**four** files: `seed_disagreement.py`, `boundary_exposure.py`, `figure_label_ceiling.py`,
`scripts/figures/boundary_limited_error.py`. Measured from the GeoTIFF transforms with a geodesic
(`pyproj.Geod`, WGS84): ireland1 x = **0.6423**, ireland2 x = **0.6354**. The repo's values are
reproduced exactly by the spherical approximation `111320·cos(phi)`; the ellipsoidal prime-vertical
radius is 1.00208x larger at 51.5N. So the shipped figures are **0.20–0.22% low**.

The same two constants `111320.0` and `111132.0` appear in **seven** files with no derivation:
`utils.py`, `build_spatial_split.py`, `block_phase_sweep.py`, `report_class_support.py`,
`accuracy_vs_separation.py`, `spatial_correlogram.py`, `assert_no_split_leakage.py`. One file uses
the wrong one of the pair: `terrain_separability.py:75` uses `111320.0` for the **latitude**
direction where every other site uses `111132.0`.

The effect is to shift the 8 m band to 8.016 m in x at the uplands and the Test B band area share
from 26.4796% to 26.430%. Small, and that is exactly why it survived — too small to notice, too
duplicated to stay consistent. It is a derivable quantity that was hand-typed five times and typed
wrong; the size of the current error is not the argument for keeping it.

**Fix:** derive the per-site GSD once from the GeoTIFF transform at run time, or define it in one
module that the other six import. Delete `gsd_for()`'s silent `(0.5, 0.5)` fallback for unknown
sites — it rescales the load-bearing band distance without saying so; it should raise.

Also hardcoded and worse: `boundary_exposure.py:74` does
`json.load(open(root/"analysis/eval_219/per_class_iou.json"))["stage1_baseline"]`. That file is
present, dated 2026-06-28, `n_tiles: 219` — the withdrawn split, leakage-inflated (Cropland IoU
0.962, fg mIoU 0.877). No tag, no provenance check. Nothing in `RUNBOOK.sh`,
`build_all_figures.py` or `docs/FIGURES.md` calls it, so it is a landmine rather than a fire.

## 4. `BIO_OEM_COMBINED` soft-defaults to the withdrawn pool

All five configs: `_BIO_OEM = os.environ.get("BIO_OEM_COMBINED", "data/biodiversity_oem_combined")`,
sitting beside a hard `_BIO_SPLIT = os.environ["BIO_SPLIT"]`. That default pool's train set contains
**239 of 294** Test A tiles, **153 of 191** Test B tiles and **141 of 173** val tiles.

Both launchers do export it, so this cannot fire today. It is one deleted line from pre-training the
transfer arm on most of both test sets and inflating precisely the contrast the paper measures.
Make it required, like `BIO_SPLIT`. Same for `compute_metrics.py`'s `--data-root` default, which is
`data/biodiversity_split/val`.

## 5a. SETTLED — why this question kept reopening, and the decision that closes it

**This had been re-decided about ten times, each pass changing the prescription.** The cause was not
indecision. It was that `spatial_blocks` returns ONE number doing TWO structurally different jobs,
and every pass argued about it as one. Traced through every consumer in the repository:

| | **Use A — description** | **Use B — inference** |
|---|---|---|
| question | does this class occur in enough distinct places? | are these exchangeable independent draws? |
| code | `MIN_CLASS_BLOCKS`, `class_block_support`, `report_class_support`, `block_size_sensitivity`, `block_phase_sweep` | `resample_blocks` in `boundary_rate_ratio`, `accuracy_vs_separation`, `bootstrap_metrics` |
| what the area argument does to it | **very little** — "Cropland occurs in 4 separated 950 m cells" is a true statement about spread whether or not those cells are full | **kills it** — you cannot resample 14 exchangeable units from 5.72 cells' worth of ground |

**Use A: keep the counts, delete the independence claim.** Write "N grid cells containing the class",
never "N independent 950 m blocks". This is a wording change and nothing else. The whole 12-vs-13-vs-14
argument dissolves, because the number stops asserting the thing it cannot support. One
implementation now exists (`report_class_support.tile_geometry` delegates to `spatial_blocks`).

**Use B: the paper's uncertainty is already per-seed, and it is not the block bootstrap.**
`aggregate_seeds.py` produces `ablation_<split>.tex`, `figure10_iou_<split>.csv` and `summary.json`
— the manuscript's tables — from **within-seed paired contrasts across the ten seeds with a paired-t
interval**. `boundary_trimap_iou.py` already writes `per_seed_class_iou_mean` and
`per_seed_class_iou_std` per radius, and its own comment records that the headline panel uses the
per-seed mean rather than the ensemble argmax. The block bootstrap in `bootstrap_metrics.py` and
`boundary_rate_ratio.py` is a SECOND, parallel uncertainty channel bolted alongside it, and it is the
one that has consumed every one of those ten iterations.

**Decision.** The reported uncertainty is over TRAINING RUNS, per seed, paired across cells. One
estimator, identical for Test A and Test B, no per-site special-casing, no merged parcels, no
jackknife-t, no phase sweep as a robustness result. It is already built, already in the pipeline,
already the table source. The block bootstrap is demoted to a single clearly-labelled sensitivity
line, or dropped.

**What this deliberately rejects.** The design panel's alternative — no interval on Test B, Test A
merged to eight along-strip parcels with a delete-one-parcel jackknife-*t* on log, per-site strip
plots. It is defensible, and it is more machinery aimed at an estimator that is not the paper's
headline. The brief for this paper is robust, not convoluted.

**What it costs, stated plainly.** The paper cannot say "this is how the number would move on new
ground". It says "this is how it moves across training runs on this ground", which is what two
purposively chosen upland sites support anyway. Nothing is claimed about spatial generalisation that
a spatial interval would have licensed, so nothing is lost that was defensible.

**Consequence for §6 and §7 below:** both were written to argue about Use B's block count. They stay
as the record of the measurement, but the estimator they describe is no longer the headline. The
measurement below is still correct and still worth stating — it is the reason the block bootstrap is
not the headline.

## 5b. The measurement — the block count overstates independence on BOTH test sets

**Corrected 2026-07-26 after a second reader checked this section.** My first version framed this as
a Test B problem and recommended reporting Kish n_eff. Both were wrong: Test A has the same defect at
almost the same magnitude, and n_eff does not close the gap. Corrected below, with the original
error left visible.

`spatial_blocks` counts **grid cells touched**, not independent parcels. Union of tile footprints,
rasterised at 2 m and reprojected to EPSG:32629:

| | covered ground | area-equivalent blocks | counted | inflation |
|---|---|---|---|---|
| Test A (inland strip) | 6.783 km² | 7.52 | **16** | **2.13x** |
| Test B (two uplands) | 5.182 km² | 5.74 | **14** | **2.44x** |
| — ireland1 | 1.829 km² | 2.03 | 7 | |
| — ireland2 | 3.354 km² | 3.72 | 7 | |

You cannot fit 16 disjoint 950 m parcels into 7.5 parcels' worth of ground, and Test A is the split
the paper leads on. Test A is a strip **1024 m wide = 1.08 cells**, so the grid cuts it into two
columns, one of them a sliver, times eight rows. ireland1's full extent is **1.57 x 1.53 cells**
(my first draft said 1.21 x 1.25, which was the span of tile *centres*, not the footprint extent; a
second reader's 1.72 x 1.53 mistook ireland2's x-span for ireland1's — the figure is 1.57 x 1.53).

**Kish n_eff is not the fix.** n_eff measures unevenness in block *size*; it does not measure
dependence between adjacent slivers. Two 13%-occupancy cells either side of a grid line are metres
apart and Kish counts them as two units. Test A: n_eff 9.85 against 7.52 of ground. Test B: 7.15
against 5.74. It moves in the right direction and stops short.

**Covered ground ÷ block area is the defensible number.** It is a genuine upper bound on how many
disjoint 950 m parcels can exist, it is derived rather than chosen, and it needs no threshold — which
matters in a repository whose repeated failure is inventing bars.

**The call, requiring no re-cut, no re-run and no new machinery:**

1. Report **all three** wherever a block bootstrap is reported, for **both** test sets: cells
   resampled (16 / 14), Kish n_eff (9.85 / 7.15), and area-equivalent blocks (7.52 / 5.74).
2. Never write "N independent 950 m blocks". Write "N grid cells resampled, spanning X km² =
   Y block-areas of ground".
3. Commit the area measurement as an artefact, like every other number in METHODS.
4. Fix `interval_coverage.py:8`, whose docstring — the stated provenance for METHODS §7 — says
   "Test A has 16 independent 950 m blocks and Test B has 14". Both halves are wrong.

**What this costs: a sentence, not a claim.** D18 retired the rho threshold, so nothing is judged on
a lower bound and an under-covering interval no longer converts into a false pass. Note in passing
that METHODS §7's operating-characteristic analysis ("the true ratio must be about 5.2 on Test A and
6.3 on Test B for a lower bound to clear 4.0") is **vestigial** — it computes the power of a
threshold that no longer exists. It should be reframed as interval coverage or cut.

The paper's second arm is barely touched by any of this: "near-boundary rate stays flat across the
four cells while the interior rate falls" is a **paired** contrast on identical ground, where every
cell sees the same pixels and the spatial dependence largely cancels. That is the ceiling argument,
and it is the part least exposed to how the ground is parcelled.

Separately, the 12-vs-14 fix was real — `spatial_blocks` and `support_blocks` now use identical
per-site scaling and both return 14 — but they agree because they are two copies of one convention.
An independent projected partition gives **13** (n_eff 8.01); the phase sweep spans 10–14. METHODS
§6's "both now give 14" is consistency, not verification.

Note also that `SPLITS = ("train","val","test")` in `build_spatial_split.py:59`, so `MIN_CLASS_BLOCKS`
**never applies to external_test**. Test B Cropland has 4 blocks and Settlement 5, both under the
8-block test floor. METHODS §4's table omits external_test. State that the floor does not cover
Test B, and why.

## 6. The two dimension-1 disagreements, and their single cause

Everything else matched to the pixel — per-class pixel counts, tile counts, train/val/test block
counts, Kish n_eff for three of four splits (27.46 / 7.52 / 9.85, exact), the 2.00x gradient-step
ratio, every OEM class share, the 2,143-tile pool, the boundary-free tile identities, Test A's
band denominator to fifteen significant figures.

The two that differ:

| | mine | repo |
|---|---|---|
| Test B blocks / n_eff | 13 / 8.01 | 14 / 7.15 |
| Test B band area share | 26.430% | 26.4796% |

Both are the metres↔degrees conversion at the two upland sites — §3's constants for the share, the
grid anchor for the blocks. No third cause exists.

## 7. Smaller, still real

- **`RUNBOOK.sh:442`** tells the operator that classes under 5 blocks "are reported as unestimable,
  never as an estimate". Those verdict labels were deliberately removed under D17
  (`report_class_support.py:55-59`). Running it prints Test B Cropland at 4 blocks unmarked. The
  removal is right; the claim is stale, and it advertises a safeguard for the one case that occurs.
- **Two comments carry the withdrawn block count**: `# 4 of 12` at `bootstrap_metrics.py:114` and
  "Test B cropland in 4 of 12" at `geoseg/utils/metric.py:98`. Should be 4 of 14 by the repo's own
  corrected function. Written in the same session that corrected METHODS §6.
- **`stage3_clsbal.py:166`** — "matching stages 1/2a/2b at 536 steps per epoch". True for 1 and 2b;
  stage 2a is **1595** steps/epoch over 3190 tiles.
- **`boundary_rate_ratio.pred_for_tile`** uses a bare `except Exception: return None`, swallowing
  missing seeds, truncated `.npy` and shape mismatches alike. Total failure is visible; *partial*
  failure silently computes rho on a subset with only the printed tile count as a clue.
- **The A0 taxonomy guard is outside the campaign window** (B4..C5), so `campaign.slurm`'s preflight
  omits it. Low risk — artefacts are tracked and commit-pinned — but it is one line beside the
  leakage gate.
- **The train split has one boundary-free tile**, `biodiversity_0808`, documented nowhere. The
  exclusion rule is stated only for the 19 upland tiles.
- **CLAUDE.md open item 1 is resolved**: the f1 confusion npz is fitted on 1072 tiles = the training
  set, and the taxonomy check passes. The "1,846 tiles" note is stale.
- **CLAUDE.md's "650 m"** is the requested buffer; the realised val–test separation is **768 m**
  (650 rounded up to the 128 m stride), which is what the gate prints.
- **METHODS §10's annotation-consistency sentence is a tautology.** "5,898,240 co-labelled pixels
  across 60 overlapping tile pairs are 100.0000% identical" — tiles are chipped from one raster on a
  50% stride, so overlapping regions are the same pixels of the same raster. Identity is guaranteed
  by the chipping. The conclusion drawn is right; the sentence must not appear as a measurement.
- **`assert_no_split_leakage.py:348`** appends the geometry line to `checks` unconditionally, so a
  failing run prints `ok geometry: ... 10 cross-split overlaps` alongside the FAIL. The exit code is
  correct, but a gate that prints "ok" next to the number that failed it is a gate people learn to
  skim.
- **The 950 m grid anchor is implicit and inconsistent across sites.** Block membership is
  `floor(x / cell)`, so the origin is whatever the CRS happens to use: the UTM 29N false easting
  (500,000 m) for the inland site, and Greenwich/the equator for the uplands. Two unrelated arbitrary
  origins, neither chosen, neither written down. METHODS §5 declares the *phase family* and the
  sweep, which is the settled part and is not reopened here — the gap is that the shipped phase is a
  side effect of the projection rather than a decision. Make the anchor a named constant with a
  stated rationale so the shipped partition is reproducible and reviewable independently of the CRS.

## 8. Not verified

No trained weights exist on this split, so no accuracy, contrast, rho, trimap curve or figure could
be checked — only the paths, with absent or planted inputs. `interval_coverage.json`'s Monte-Carlo
values were not re-derived; its Test B inputs inherit the 14-block convention §5 disputes. The
literature in METHODS §10/§11 was not opened, so its own caveats stand (Cheng read from arXiv not
the CVPR proceedings and absent from `Bibliography.bib`; Liu et al. 2016 paywalled). The OEM
relabelling of the 2,118 pool tiles was not re-derived from the teacher confusion. Sonic was not
touched; `campaign.slurm` has never run.

## 9. The pattern worth naming

Attack 2 succeeded against a guard written in the previous session **specifically to stop it**,
because the guard validates the checkpoint and not the data. Every provenance check in the tree
validates `checkpoint` and `tta`; none validates `split` or `data_root`, and both are already sitting
in the file. The recurring shape is not "no check" — it is a check drawn in the same frame as the
thing it checks. Two of the three ledger rows I re-tested (C5, C7) are marked DONE and are not.
