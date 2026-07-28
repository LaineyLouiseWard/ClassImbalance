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

IoU with no exclusion, and after excluding 8 px (4 m) around every boundary. **Per seed, not from
the ensemble**, read out of the JSON rather than the console table:

| class | baseline IoU | boundaries excluded | gain |
|---|---|---|---|
| Settlement | 0.717 | 0.890 | **+17.3 pp** |
| Forest | 0.716 | 0.847 | **+13.2 pp** |
| Grassland | 0.844 | 0.889 | +4.6 pp |
| Cropland | 0.460 | 0.505 | +4.5 pp |
| **Seminatural** | 0.414 | 0.449 | **+3.5 pp** |
| macro foreground | 0.630 | 0.716 | +8.6 pp |

**Two of the four caveats below are already discharged by this table.** The console prints only three
classes, but the JSON carries all five at every radius, per seed, with standard deviations — so no
rerun is needed to see grassland and cropland. And the per-seed figures match the ensemble ones to
within 0.1 pp (13.2 against 13.3, 17.3 against 17.2, 3.5 against 3.4), so the ensemble was not
inflating this particular comparison.

## What it means

**Two classes fail at their edges. Three do not.** Settlement and forest recover 13 to 17 points once
the boundary band is removed. Grassland, cropland and semi-natural recover 3 to 5 points and stay
where they were — semi-natural still scores 0.449 with every pixel within 4 m of a boundary thrown
away.

The split is not "the weak classes versus the strong ones". It is **the classes with a physical edge
versus the classes without one.** A settlement or a forest has a real perimeter you can trace. A field
boundary between improved and semi-natural grassland is a management gradient, and cropland against
grassland is a decision about what a field currently holds. For those three the error is inside the
parcel.

That is whole-parcel misassignment: the outline was drawn correctly and the field was called the
wrong type — or the field genuinely could be either. It is not an edge-precision problem, and no
amount of more careful tracing addresses it.

## The stronger evidence: error rate against distance, per class

The trimap gain is one summary of a curve. The curve itself is in the same JSON, thirteen distance
bins per class. Error rate, with foreground pixel support beside it:

| class | 0-0.5 m | 1-2 m | 4-6 m | 8-12 m | 16-24 m | 32 m+ |
|---|---|---|---|---|---|---|
| Forest | 39.1% *0.8M* | 20.7% *1.0M* | 5.7% *1.5M* | 4.9% *0.8M* | 7.9% *0.4M* | **0.9%** *0.5M* |
| Settlement | 44.5% *0.3M* | 21.9% *0.3M* | 4.8% *0.3M* | 5.4% *0.3M* | 2.8% *58k* | **0.0%** *2k* |
| Grassland | 55.9% *0.9M* | 29.4% *1.1M* | 11.2% *3.1M* | 9.1% *5.1M* | 8.2% *7.2M* | **7.1%** *14.0M* |
| Seminatural | 58.8% *86k* | 52.0% *0.1M* | 36.9% *0.3M* | 28.6% *0.5M* | 24.3% *0.6M* | **13.5%** *1.7M* |
| Cropland | 77.1% *25k* | 61.7% *32k* | 47.5% *82k* | 41.5% *0.1M* | 36.5% *0.2M* | **77.5%** *98k* |

**Read the last column.** Forest falls to 0.9% and settlement to 0.0% — those two really do collapse
to a near-zero interior. Grassland holds a 7.1% floor across fourteen million pixels. Semi-natural
holds **13.5%** across 1.7 million. Cropland has no boundary structure at all: 77% at the edge, 37%
in the middle distance, 77% again deep inside.

**And for semi-natural the interior is where the pixels are.** Only 86,000 of its pixels sit in the
first half-metre; 1.7 million sit beyond 32 m. So most of its error is interior error by mass, not
just by rate.

This is the same conclusion as the trimap table, measured a different way, and it is the version to
put in the paper because it shows the shape rather than one summary of it.

**It also settles a sentence `DO_NOT_ADD.md` already forbids.** "Every class collapses to a near-zero
interior rate" is now not merely unsupported but demonstrably false for three of the five classes.
The honest version is that two classes collapse and three do not, and which is which is the finding.

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
2. ~~Print all five classes~~ — **DISCHARGED.** The JSON already carries all five per seed at every
   radius. Only the console table is limited to three, which is cosmetic.
3. ~~Per-seed rather than ensemble~~ — **DISCHARGED.** The two agree to 0.1 pp on this comparison.
4. **Restate Q3 as a per-class result.** "Error concentrates at class boundaries" as a global claim
   does not survive this if it holds.
5. **Check the per-class error-versus-distance curves**, which the same JSON carries at thirteen
   distance bins per class. The trimap gain is one summary of those curves; the curves themselves say
   whether semi-natural's error is flat with distance or merely less steep.

## What does not change either way

The class-pair share (47%, area-weighted 2.1x), the bounded nulls on both interventions, the sampler
acting and buying almost nothing, and the stability of the near-boundary error rate across all forty
runs. Those are measured on different quantities and are untouched by this.
