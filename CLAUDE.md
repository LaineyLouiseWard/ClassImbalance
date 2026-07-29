# CLAUDE.md — Project Instructions

## What this repository is

> **The label-quality-ceiling claim that stood here was withdrawn 2026-07-29.** Current narrative:
> **`notes/NARRATIVE_LITERATURE_FINAL.md`** (supersedes `docs/NARRATIVE_FINAL.md`). Rebuild plan:
> `notes/IMPLEMENTATION_PLAN.md`. **Not submitted** — no manuscript ID, no submission date; nothing is
> locked by a prior version.

Case study being prepared for *Remote Sensing*, special issue "Advancing Earth Observation Through
Artificial Intelligence", topic *Data Curation for AI*.

On a fixed FT-UNetFormer, two data-curation interventions — OpenEarthMap pre-training and minority
oversampling — in a 2×2 factorial over ten seeds. **Neither main effect separates from run-to-run
variation**; the design resolves about 3 pp of foreground mean IoU, wider than the spread between cells.

46.7% of foreground error is Grassland and Seminatural confused **with each other**, both directions
pooled — more than twice the share their ground area alone would give them. Holding the source class
fixed and starting from grassland pixels: the model is about 15× likelier to call grassland forest
within 8 m of a class boundary than beyond it (per seed 9.8–20.2), while its rate of calling grassland
semi-natural is flat with distance (0.86, per seed 0.59–1.13). **The bare shares — 75.3% and 15.0%
beyond 8 m — must not lead:** 72.4% of grassland's own ground already lies beyond 8 m, so 75.3% is
1.04× that background and is not a finding. **Never compare across source classes** — object size
confounds it, and that comparison is withdrawn. Model
failure, parcel-level label error and majority absorption all remain open. **The paper locates the
error and stops.**

- Manuscript: `manuscript/main.tex` (compiles with `latexmk`). Cover letter: `manuscript/cover_letter.tex`.
- Goal for the code: a **clean, reproducible mirror** of the paper. Do not change scientific results,
  model architecture, training logic, or evaluation procedures — only organisation, reproducibility,
  and the write-up.

## Reproducing

`RUNBOOK.sh` is the single entry point for the full pipeline: data prep, OpenEarthMap taxonomy
grounding, training the four factorial cells over ten seeds, evaluation, and figures. `RUNBOOK.md`
documents each stage and the stage→config map; use `--from <STAGE>` to resume.

```bash
bash RUNBOOK.sh                                # full pipeline from scratch
python scripts/figures/build_all_figures.py    # rebuild the paper figures
```

## The 2×2 factorial

A fixed FT-UNetFormer baseline crossed over two off/on factors — (A) OpenEarthMap transfer and
(B) the class-balanced (clsbal) sampler — gives four cells, each trained over ten seeds (42–51):

| Cell | Config (`config/biodiversity/`) |
|------|---------------------------------|
| baseline (both off) | `stage1_baseline.py` |
| transfer-only | `stage2b_oem_finetune.py` (from `stage2a_oem_pretrain.py`) |
| sampler-only | `stage_sampler_only.py` |
| full (**shipped model**) | `stage3_clsbal.py` (stage2a + clsbal) |

**Dropped — kept only as the Discussion's negative results, not in the pipeline:** knowledge
distillation / self-distillation (a 5-seed test gave −0.54 pp versus a step-matched no-KD control)
and the bespoke hardness×richness sampler (retired in favour of clsbal). Their code
(`train/train_kd.py`, `geoseg/losses/selfdistill.py`, `config/biodiversity/stage3_sampler.py`, and
related utilities) remains in the tree for reference but is excluded from the shipped pipeline.

## Key facts

- **Datasets:** Biodiversity (proprietary, ~9 GB, not public; ODOS Technologies) + OpenEarthMap
  (public). **Split (rebuilt 2026-07-26 after a leakage finding):** spatially blocked, one axis cut of
  the inland site — `train 1072 | 256 m | val 173 | 768 m | test 294` — plus the two upland sites held
  out whole as `external_test` (191). Built by `build_spatial_split.py`, manifest
  `artifacts/spatial_split_manifest_f1.json`. The old 1,706/219/218 random split LEAKED (50% stride
  chipping meant ~93% of each held-out tile's ground was also in training) and every number produced
  from it is withdrawn.
- **Model:** FT-UNetFormer (Transformer-backbone UNetFormer) with a Swin-Base encoder pre-trained
  ImageNet-1K → ADE20K (UPerNet), held fixed across all four cells.
- **Class order:** Background=0, Forest=1, Grassland=2, Cropland=3, Settlement=4, Seminatural=5 —
  defined in `geoseg/taxonomy.py` (`STUDENT_CLASSES`); all configs, metrics, and confusion matrices
  follow it.
- **Foreground mIoU:** `np.nanmean(iou[1:])` — the five foreground classes only (Background
  excluded). Used for checkpoint selection and every reported metric (`evaluation/compute_metrics.py`).
- **Shipped sampler (minority oversampling, `clsbal` in code):** frequency-only class-balanced
  sampling (Kang 2020), built by `scripts/data_prep/build_clsbal_sampler.py` →
  `artifacts/sampler_weights_clsbal_f1.tsv` (q=1.0, `--settlement_target 1.27`).
  **Neither 1.27 nor 2.84 may be quoted as a result** — 1.27 is the target the binary search drives
  toward, so it is circular; 2.84 is a reproducible measurement but has no ledger row. Both are ratios
  of mean tile sampling weights, so both depend on the tile list they are calibrated against — name it
  and add a ledger row before quoting either.
- **OEM taxonomy grounding:** an OEM-trained model's confusion on the Biodiversity training set
  grounds the OEM→Biodiversity class mapping (`artifacts/teacher_oem_gt_confusion.npz`, committed).
- `evaluation/evaluation_results/` is gitignored and regenerated by `RUNBOOK.sh --from C1`.

## Figures

**Seven figures** (down from thirteen; the cut list and reasons are in `docs/FIGURES.md`), built by
`python scripts/figures/build_all_figures.py`. The authoritative figure map (content → script →
output) is **[docs/FIGURES.md](docs/FIGURES.md)** — keep it in sync when figures change. Figures use
stable **descriptive** names (e.g. `boundary_limited_error.pdf`); LaTeX assigns the printed numbers
via `\ref`. The graphical abstract is built separately
(`scripts/figures/graphical_abstract_panels.py` + `graphical_abstract_tikz.tex`).

## Working style — written from what actually went wrong here

**Be short.** Chat replies of a few sentences, not a few screens. A long reply reads as less decisive,
not more thorough, and it costs reading time that does not exist near a deadline. Lead with the answer.
No preamble, no restating the question, no closing summary of what you just said.

**Mutation-test every gate.** Anthropic's Opus 5 guidance says to REMOVE generic "verify your work"
instructions — the model already self-verifies and they cause over-verification. The failure seen here
is narrower and real: on 2026-07-28 every gate written passed first time and four could not fail. The
tile-edge guard lived in `main()` while the self-test only called helpers; a contact test was blind to
one axis because the fixture's only seam ran the other way; a denominator fix was asserted by a line
that recomputed it rather than calling it. So: break the thing the gate guards, watch it fail, revert.
See "Gate discipline" below.

**Use adversarial subagents on this project, despite the general guidance against it.** Anthropic's
Opus 5 notes say not to delegate verification. Here it earns its cost: on 2026-07-28 independent
checkers refuted a claim about which direction the class confusion ran, found a contact denominator
wrong by a factor of two, and identified four gates that could not fail. The pattern is that the first
interpretation of a result is usually wrong, and a checker that re-derives from raw data rather than
reading the code is what catches it. Give them a different path to the answer, not the same one.

**Quote before asserting.** If you say a document says something, quote the line. If you say a number
reproduces, run `scripts/analysis/verify_narrative_numbers.py`. Several claims that session were
confidently wrong about files open in the same conversation.

**When a check fails, suspect the check first.** Four failures that session were wrong expectations,
not wrong code — a band computed in pixels where the code used metres, a symmetry counted twice, an
inverted distance transform. Read the failure before changing the thing under test.

## Conventions

- Paths are repo-root-relative; run commands from the repo root without `cd`. No hard-coded absolute
  paths (figure scripts use `find_repo_root()`).
- Figure outputs go flat in `figures/`; their scripts live in `scripts/figures/` or `scripts/analysis/`.
- **Archiving uses one place: `../_archive-lqc/`, a sibling of this repo** (cut figures under
  `../_archive-lqc/cut_figures/`). It sat inside the repo as `_archive/` until 2026-07-28 and was
  moved out at 86 GB, because gitignoring it stops it cloning but not every backup, indexer and
  recursive `find` from walking it. Do not scatter per-subfolder `_orphaned/` directories.
- **Working notes and plans live in `notes/` (gitignored, never published).** The only tracked
  documentation is `docs/DESIGN_NOTES.md` (design decisions and negative results) and
  `docs/FIGURES.md` (figure map); `README.md` and `RUNBOOK.md` are the public entry points.

## STATE AS OF 2026-07-28 — read this before touching anything

**The campaign is complete.** Four cells x ten seeds on the rebuilt split (`650561`, commit
`c5908c8`), scored on the deduplicated non-overlapping subset. `docs/RESULTS_TEN_SEED.md` §7-§13 is
the only numerical authority; §1-6 of that file are superseded where they conflict.

**Every number the narrative quotes has a committed artifact and a row in a ledger.** Run

    PYTHONPATH=. python scripts/analysis/verify_narrative_numbers.py

before quoting anything and after any recomputation, and `--markdown` regenerates `docs/NUMBERS.md`,
the readable map of which file holds which number. It checks 64 numbers against the artifacts under
`analysis/` and `artifacts/`, and for any that cannot be resolved it prints the command that
regenerates them. A number with no row there is a number nobody can defend — add the row when you add
the number.

**Still stale, by decision:** `README.md`, `RUNBOOK.md` and the top of `docs/DESIGN_NOTES.md` describe
the old split and state the old conclusions as fact. The withdrawn campaign's outputs — predictions,
checkpoints, evaluation results — were moved to `../_archive-lqc/withdrawn_campaign_2026-07-28/` on
2026-07-28, so nothing can read them by accident.

**The narrative itself** is `notes/NARRATIVE_LITERATURE_FINAL.md`, which **supersedes**
`docs/NARRATIVE_FINAL.md` — the older file predates the literature gating and quotes rho on the
294-tile population rather than the 90-chip subset that is reported. `docs/FINDING_BOUNDARY_IS_PER_CLASS.md`
carries the per-class boundary result (corrected 2026-07-29 against
`analysis/label_ceiling/test/boundary_trimap_stage1_baseline.json`; the earlier table was transcribed
from a two-seed probe with Forest and Settlement swapped). All carry the wording constraints;
`docs/DO_NOT_ADD.md` remains absolute.

The rebuild plan is `notes/IMPLEMENTATION_PLAN.md`.

### The design, in five lines

- One inland site cut along one axis: `train | 256 m | val | 768 m | test`, tiles straddling either
  buffer band dropped. Two upland sites held out whole as a second test set.
- 256 m is the exact pixel-identity distance (512 px at 0.5 m on a 128 m stride). 650 m was the
  REQUESTED val/test buffer; tiles straddling the band are dropped on a 128 m stride, so the REALISED
  separation is 768 m — that is what the leakage gate prints and what the paper must state. It
  **clears** the **750 m** measured autocorrelation range of the inland site, the one Test A is cut
  from, as does the 1,664 m train–test separation; peak Mantel r = 0.044 either way. The gap that does
  sit inside the range is train–val at 256 m, and that is where every checkpoint is selected.
  (This line said "below the 950 m range" until 2026-07-29. Both halves were wrong: 950 m is
  `ireland2`'s composition range and separately `SUPPORT_BLOCK_M`, the grid-cell size — and 768 m is
  above 750 m, not below it.)
- Splits are accepted or rejected on **950 m grid-cell support** per class, not on class share: a
  share does not track estimability (7.00% share in 3 cells is unusable; 1.91% in 11 is fine). Test
  gets a stricter minimum (8 cells) than train and val (5). Two cautions, both in METHODS §6 and §4:
  these are cells CONTAINING the class, never "independent blocks", and the criterion is applied to
  train/val/test only — `external_test` has no floor, and its Cropland sits in 4 cells.
- Two test sets, never pooled: Test A = inland strip (accuracy on new ground inside a surveyed area),
  Test B = uplands (transfer to unsurveyed terrain).
- There is ONE split. The three-fold design was retired 2026-07-26; folds were role-permutations of
  one site and could not be averaged as replicates.

### Before writing prose or commissioning a review — READ `docs/DO_NOT_ADD.md`

Sentences this design forbids, and sources that do not exist or do not say what they are cited for.
It exists so the next chat does not re-audit what has already been settled three times. Two entries
are load-bearing enough to repeat here:

- **"Ortiz et al. 2025 (TGRS)" IS FABRICATED.** Removed from the code and the manuscript, but ~40
  references survive in `notes/` presenting it as read in full. Four files now carry warning banners.
  Its "88% inter-annotator bound" is **not** the expert mask-error audit that motivates this paper —
  two different quantities share the number 88, and one external reviewer has already conflated them.
- **The 8 m band cannot be cited to Kohli.** His "8" is 8 *pixels* on 320x213 images, in a figure
  caption. Ours is 8 metres. Coincidence.

### The registered claim

`docs/audit/PREREGISTRATION_P1_AMENDMENT.md` is the authority, and it is auditable: it contains all three
versions, two of them retracted, dated before any training.

- **Primary statistic: rho** = (foreground error rate within 8 m of a GT boundary) / (rate beyond 8 m).
  Implemented in `scripts/analysis/boundary_rate_ratio.py`, with a self-test.
- **No threshold and no interval.** D18 retired the rho >= 4.0 bar, and on 2026-07-26 the block
  bootstrap that was to supply its lower bound was REMOVED: it existed only to defend that bar, both
  test sets are complete enumerations of their ground (no sample, so no sampling error), and every
  claim is either a cross-cell contrast on identical pixels or a census level. Uncertainty is
  PER-SEED AND PAIRED — `aggregate_seeds.py` plus the per-seed curves in `boundary_trimap_iou.py`.
  One estimator, not two. METHODS §7.
- **Second arm, free:** if error is label-limited, the near-boundary rate must be roughly flat across
  the four cells while the interior rate falls. The comparison is **in relative terms**: the near-boundary rate must vary by LESS than the
  interior rate does. **The falsifier, which must be stated with the claim:** if both fall
  proportionally, the concentration is a property of model quality and the label-ceiling reading
  is NOT supported by this evidence.
  rho alone cannot separate label-limited from capacity-limited, because rho rises as a model
  improves. Note the arm is a NECESSARY condition, not a diagnosis: every rival cause that is
  constant across the four cells — encoder/decoder edge blur, mixed pixels at 0.5 m, registration
  offset — predicts the same flat near-boundary rate, and the architecture is held constant in
  every contrast this design computes.
- Do NOT use the error *share* within 8 m, or that share divided by area share (which is `lift`). Both
  were registered and retracted: they are landscape-dependent, and the repository documents exactly
  why. The manuscript's 92%/96% shares are leakage-inflated and must never appear.

### Working style — written from what actually went wrong here

**Be short.** Chat replies of a few sentences, not a few screens. A long reply reads as less decisive,
not more thorough, and it costs reading time that does not exist near a deadline. Lead with the answer.
No preamble, no restating the question, no closing summary of what you just said.

**Mutation-test every gate.** Anthropic's Opus 5 guidance says to REMOVE generic "verify your work"
instructions — the model already self-verifies and they cause over-verification. The failure seen here
is narrower and real: on 2026-07-28 every gate written passed first time and four could not fail. The
tile-edge guard lived in `main()` while the self-test only called helpers; a contact test was blind to
one axis because the fixture's only seam ran the other way; a denominator fix was asserted by a line
that recomputed it rather than calling it. So: break the thing the gate guards, watch it fail, revert.
See "Gate discipline" below.

**Use adversarial subagents on this project, despite the general guidance against it.** Anthropic's
Opus 5 notes say not to delegate verification. Here it earns its cost: on 2026-07-28 independent
checkers refuted a claim about which direction the class confusion ran, found a contact denominator
wrong by a factor of two, and identified four gates that could not fail. The pattern is that the first
interpretation of a result is usually wrong, and a checker that re-derives from raw data rather than
reading the code is what catches it. Give them a different path to the answer, not the same one.

**Quote before asserting.** If you say a document says something, quote the line. If you say a number
reproduces, run `scripts/analysis/verify_narrative_numbers.py`. Several claims that session were
confidently wrong about files open in the same conversation.

**When a check fails, suspect the check first.** Four failures that session were wrong expectations,
not wrong code — a band computed in pixels where the code used metres, a symmetry counted twice, an
inverted distance transform. Read the failure before changing the thing under test.

## Conventions that are easy to get wrong

- `boundary_distance(mask, tile_id)` returns **METRES** and needs the tile id for per-site pixel size.
  The uplands are anisotropic (0.515 x 0.641 m); omitting the id silently rescales every band.
- Band membership is **strict** `< 8.0 m`.
- Tiles with no GT boundary are **excluded** from rho: 19 of 191 upland tiles (all `ireland2`),
  and one in train (`biodiversity_0808`, 0.09% of train foreground); 16% of Test B
  foreground. Including them moves Test B's band area share from 26.480% to 22.199%.
- The registered denominators live in `artifacts/boundary_band_denominators.json`, regenerated by
  `scripts/analysis/register_boundary_denominators.py`. Never hand-edit.
- `SPLIT_TAG` gates every checkpoint and evaluation path and **must be exported**. The untagged paths
  still exist under `../_archive-lqc/stale_checkpoints_pre_rebuild/` and would score the withdrawn campaign.
- `max_epoch = 45`, not 50: `T_0=15, T_mult=2` means cycles end at 15, 45, 105, so training must stop
  at one of those to end at an LR minimum.

### Gate discipline — the lesson this repo keeps teaching

Roughly fifteen defects were found across three reviews, and the recurring shape is **a check written
in the same frame as the thing it checks, never run against a known-bad input**. Six gates could not
fail. Three constraints sat on code paths the shipped command never reached. So:

> **A gate that has not been observed to fail does not exist.** Construct the input it should reject
> and watch it reject it. Trace every constraint from its constant to the line that actually runs.

### What is still open

1. ~~The teacher confusion matches no split.~~ **RESOLVED 2026-07-26.**
   `artifacts/teacher_oem_gt_confusion_f1.npz` is fitted on 1,072 tiles — exactly the training set,
   confirmed by the leakage gate — and `verify_taxonomy_consistency.py` passes 31 checks with no
   drift against the hard-coded `taxonomy.OEM_TO_STUDENT_PRETRAIN`. The untagged 1,846-tile file
   belongs to the withdrawn campaign. The check now also runs in the cluster preflight, since the
   campaign window B4..C5 excludes stage A0.
2. Convergence at 1,072 training tiles is assumed, not shown. The budget was set for 1,706. Check the
   validation curve from the first completed run before spending the rest.
3. Campaign: 4 cells x 10 seeds = 40 runs on one split. `RUNBOOK.sh` stages A0..E; the fourth cell is
   B4c and Test B scoring is C1b, both added 2026-07-26.
4. `sonic/10_submit_final_campaign.slurm` bypasses the preflight and tests for withdrawn artefacts.
   Do not reuse it; write a new one under `sonic/campaign/` (the only tracked path there).
5. No cut-placement sensitivity is measurable with one split. State as a limitation.
6. Manuscript: §2 describes a split that does not exist. Largest remaining task.
