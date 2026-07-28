# The boundary result is per-class, and the weak classes do not take part

**Provisional, 2026-07-28. Two seeds, one cell, Test A. The ten-seed run over four cells and both
test sets is in progress and will confirm or kill this.** Written now because if it holds it changes
which claim the paper leads on.

## Nothing is wrong with anything already computed

No number in `RESULTS_TEN_SEED.md` changes. No arithmetic was found to be wrong. This is a
**measurement nobody had run on the rebuilt split**, not a correction to one that had been.

## What was run, exactly

`scripts/analysis/boundary_trimap_iou.py`, on the clean campaign, for the first time. It was run as a
validation probe before launching the full Stage 1 sweep — two seeds (42, 43), the baseline cell,
Test A, on a fresh checkout of the current commit with the data symlinked in. About 110 seconds.

    PYTHONPATH=. python scripts/analysis/boundary_trimap_iou.py \
        --softmax-root $SONIC/results/softmax --mask-dir data/split_f1/test/masks \
        --cell stage1_baseline --seeds 42 43 --out-dir <out>

**What the script does.** It rescores per-class IoU while progressively excluding a band of pixels
around every ground-truth class boundary. If a class's errors sit at its edges, its IoU rises sharply
as the band widens. If its errors sit inside parcels, the IoU barely moves. This is the standard
trimap or boundary-band diagnostic (Kohli et al. 2009; Csurka et al. 2013), read in the
exclude-the-band direction.

## The result

IoU with no exclusion, and after excluding 8 px (4 m) around every boundary:

| class | baseline IoU | boundaries excluded | gain |
|---|---|---|---|
| Settlement | 0.723 | 0.895 | **+17.2 pp** |
| Forest | 0.721 | 0.854 | **+13.3 pp** |
| **Seminatural** | 0.421 | 0.455 | **+3.4 pp** |
| macro foreground | 0.632 | 0.716 | +8.3 pp |

Grassland and Cropland are not in the printed table — the script prints only the three classes
flagged as narrative-focus in `HARD`. **The full run must print all five**, and that is a change to
make before quoting any of this.

## What it means

**Forest and settlement fail at their edges. Semi-natural grassland does not.** Remove every pixel
within 4 m of a class boundary and semi-natural still scores 0.455. Its error is inside parcels.

That is whole-parcel misassignment: the outline was drawn correctly and the field was called the
wrong type — or the field genuinely could be either. It is not an edge-precision problem, and no
amount of more careful tracing addresses it.

## Why this is uncomfortable, stated plainly

The narrative currently leads on boundary concentration. If the boundary concentration is carried by
forest and settlement — the two classes that are **already segmented well** — then the headline
result is about the classes nobody has a problem with, while the pair carrying **47% of all
foreground error** fails somewhere the headline does not describe.

A referee who runs this decomposition, which is one command against outputs the paper already
reports, reaches that in an afternoon. Better to lead with it than to be shown it.

## Why it is probably better for the paper, not worse

It splits one vague finding into two specific ones, and the second is the more useful:

- **Classes with real edges** (forest, settlement): error is at the edges, and better tracing or finer
  imagery plausibly reduces it.
- **The grassland pair** (47% of foreground error): error is inside parcels. Boundary annotation
  recovers essentially nothing there.

That converts the paper's practical recommendation from a hedge into an instruction — it says where
annotation money is wasted, and why. It also moves the finding closer to a data-curation contribution
than a boundary-metrics one, because whole-parcel class ambiguity is a question about the class
definitions rather than about tracing.

It is also what `notes/rebuild_2026-07/for_the_paper/NARRATIVE_OPTIONS.md` predicted before any of
this was run, as candidate N2: boundary labelling buys a real, bounded improvement and then stops at
a ceiling set by parcel-level class ambiguity. That note called N2 "most likely on current evidence".

## What has to happen before any of this is written into the manuscript

1. **The ten-seed, four-cell, both-test-set run** now in progress. Two seeds is not a result.
2. **Print all five classes**, not the three in `HARD`. Grassland and cropland are missing and
   cropland is the other weak class.
3. **Per-seed rather than ensemble.** The printed table is the ensemble argmax, which removes interior
   error preferentially and therefore flatters exactly the quantity being compared here. The script
   already computes a per-seed version; that is the one to quote.
4. **Restate Q3 as a per-class result.** "Error concentrates at class boundaries" as a global claim
   does not survive this if it holds.

## What does not change either way

The class-pair share (47%, area-weighted 2.1x), the bounded nulls on both interventions, the sampler
acting and buying almost nothing, and the stability of the near-boundary error rate across all forty
runs. Those are measured on different quantities and are untouched by this.
