# Plan — the coding left before submission

Written 2026-07-28. No new analysis, no retraining, no new metrics. Everything here is running code
that already exists against the campaign that already finished.

## Why four numbers are missing from the repository

The deduplicated rho (2.14 at 8 m), the area-weighted pair ratio (2.1x against 0.6x), the per-class
paired confidence intervals, and the seed-only control were all computed during a chat session — SSH
one-liners on the cluster and throwaway Python in a scratch directory — rather than through repository
scripts. The numbers are right and were checked, but they are not reproducible by anyone else, and
`RUNBOOK.sh` does not produce them. Only the seed control has been written up so far
(`RESULTS_TEN_SEED.md` §10). **Stage 4 fixes this and it is not optional: a number in the paper that
no script produces is a number the author cannot defend.**

## What does NOT need to move

`$SONIC/results/softmax` is 78 GB and `checkpoints` is 58 GB. Neither is needed locally. Every
surviving figure is built from a derived artifact measured in kilobytes. Run the analysis on the
cluster, bring back JSON.

The one exception is the qualitative panel, which needs the raw dumps for a handful of named tiles —
about 40 files, a few hundred megabytes at most.

## Stage 1 — on Sonic, produce the derived artifacts (~1 h, mostly waiting)

All four scripts already exist and take the campaign outputs as input.

| what | script | output |
|---|---|---|
| per-cell, per-seed metrics already fetched | — | `results/{test,external,val}/*.json`, 40 each, already local |
| boundary curve and trimap recovery | `scripts/analysis/boundary_trimap_iou.py` | one JSON per cell |
| rho, per seed, both widths, both test sets | `scripts/analysis/boundary_rate_ratio.py` | one JSON per cell |
| confusion matrices, ten seeds pooled, per cell | `evaluation/compute_metrics.py` (already writes them) | confusion CSV per cell |

Two things to get right, both already fixed in the code but not yet exercised on the campaign:

- Run everything with `--split-root` pointing at the **deduplicated scoring subset** so rho and the
  IoU table are censuses of the same tile population. This is the asymmetry `RESULTS_TEN_SEED.md`
  §12 flags and it must be resolved one way and stated.
- `boundary_trimap_iou.py` now excludes boundary-free tiles, which changes Test B interior rates by
  1.26x against anything computed before today.

## Stage 2 — fetch (~10 min)

`rsync` the derived JSONs and confusion matrices into `analysis/` and `evaluation/evaluation_results/`.
Then the ~40 softmax files for the tiles the qualitative panel uses. **Pick those tiles from
`data/split_f1/test`** — the current defaults are two tiles that are now in train and test
respectively, and one of them would put a training tile in a manuscript figure.

## Stage 3 — rebuild only the figures that survive (~1-2 h)

Seven of thirteen. `build_all_figures.py --skip` takes the rest out.

**Keep, and what each now carries:**

| figure | carries | data needed |
|---|---|---|
| workflow pipeline | the 2x2 as a control | none (TikZ) |
| study area | the blocked split and the buffers | split manifest, already local |
| class distributions | the imbalance that motivates Q1 | `class_support.json`, already local |
| OpenEarthMap mapping | why pre-training is not an imbalance remedy for two classes | committed artifact; **must be rebuilt on the regrounded mapping** — the current PDF draws the pre-regrounding version |
| confusion matrices | **Q2** — the pair, on absolute volumes | confusion CSV from Stage 1 |
| boundary-limited error | **Q3 and Q4** — the ratio at two widths, and its stability | trimap JSON from Stage 1 |
| qualitative panel | repurposed: what the two grasslands look like side by side | softmax for a few Test A tiles |

**Cut, six figures and roughly five pages:**

frequency vs difficulty (the claim is unsupportable in either direction at n=5) · reliability/ECE ·
uncertainty quality · uncertainty overlay · confident-learning appendix figure · the two-axes
mitigation schematic (it draws a data-versus-model split the design cannot test).

## Stage 4 — make the four session-only numbers reproducible (~1 h)

One script, `scripts/analysis/narrative_numbers.py`, writing one JSON, so every number in the abstract
and conclusions has a command behind it:

1. **rho on the deduplicated subset** — already supported by `boundary_rate_ratio.py` via
   `--split-root`; just needs running and recording rather than an ad hoc symlink tree.
2. **Area-weighted class-pair ratio** — for each pair, its share of foreground error divided by the
   product of the two classes' foreground area shares. Gives 2.1x for the grassland pair and 0.6x for
   forest-grassland. This is the statistic Q2 now rests on and it exists nowhere in the repository.
3. **Per-class paired contrasts with intervals** — `aggregate_seeds.py` already computes paired
   contrasts on the aggregate; extend the same function over the five per-class IoUs. Needed because
   `DO_NOT_ADD.md` forbids calling anything a null without a stated bar.
4. **The seed-only control** — CV across seeds within a cell against CV across cells within a seed,
   for both band widths. Already written up; needs the code behind it.

Add a self-test to each, in the style of `boundary_rate_ratio.py`'s — and check it fails when
mutated, because this repository's rule is that a gate never seen to fail does not exist.

## Order, and the cut line

1. Stage 1 and 2 first — everything downstream waits on them.
2. Stage 4 next, not last. If the area-weighted ratio comes out differently from 2.1x, Q2's wording
   changes and that changes the abstract.
3. Stage 3 last, because a figure built before its numbers are settled gets built twice.

**If time runs short, cut in this order:** the qualitative panel (nice, not load-bearing), then the
OpenEarthMap mapping rebuild (demote the figure to a table), then Stage 4's self-tests — but never
Stage 4 itself, and never the confusion or boundary figures, which are Q2, Q3 and Q4.
