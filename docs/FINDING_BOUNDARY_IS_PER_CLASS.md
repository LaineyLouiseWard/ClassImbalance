# The weak classes fail inside parcels, not at boundaries

**CONFIRMED 2026-07-28 on ten seeds, four cells, both test sets** (Slurm 652544). The provisional
two-seed version below stands, with one correction that matters: **the cross-class comparison it
originally made is confounded by object size and has been withdrawn.** What survives is a set of
within-class statements, and they are stronger than what was originally claimed.

## The confirmed result

Error rate on pixels lying **more than 32 m from any class boundary**, ten-seed ensemble, with pixel
support:

**Test A** — 294 tiles, no boundary-free tiles

| cell | Forest | Grassland | Cropland | Seminatural |
|---|---|---|---|---|
| baseline | 0.6% *0.5M* | 5.1% *14.0M* | 78.8% *98k* | **18.8%** *1.7M* |
| transfer only | 0.9% *0.5M* | 5.0% *14.0M* | 76.0% *98k* | **25.7%** *1.7M* |
| sampler only | 0.9% *0.5M* | 5.0% *14.0M* | 78.0% *98k* | **28.9%** *1.7M* |
| full | 0.9% *0.5M* | 6.1% *14.0M* | 81.2% *98k* | **24.3%** *1.7M* |

**Test B** — 172 tiles after excluding 19 boundary-free

| cell | Forest | Grassland | Cropland | Seminatural |
|---|---|---|---|---|
| baseline | 25.9% *0.1M* | 1.4% *1.1M* | 0.1% *0.4M* | **26.0%** *10.1M* |
| transfer only | 24.4% *0.1M* | 2.0% *1.1M* | 63.1% *0.4M* | **42.8%** *10.1M* |
| sampler only | 22.5% *0.1M* | 2.0% *1.1M* | 0.3% *0.4M* | **29.8%** *10.1M* |
| full | 28.8% *0.1M* | 0.9% *1.1M* | 44.1% *0.4M* | **22.8%** *10.1M* |

**Settlement's deep-interior figure is noise, wherever it appears.** It has about 2,900 pixels beyond
32 m on Test A and fewer on Test B — see the object-size confound below. It now appears in the
per-class table further down at 0.0%, and the paragraph under that table reads it alongside forest's
0.6%. **Settlement's 0.0% must not be quoted in the paper** — 2,900 pixels cannot support it. Forest's
0.6%, on 475,555 pixels, can.

**Three things this says.**

1. **Semi-natural grassland fails deep inside parcels.** 18.8% of its pixels more than 32 m from any
   boundary are wrong on Test A, across 1.7 million pixels; 26.0% across 10.1 million on Test B. This
   cannot be a boundary-placement problem, because these pixels are nowhere near a boundary.
2. **Cropland does the same, worse.** 78.8% beyond 32 m on Test A. It has no boundary structure at
   all.
3. **Neither intervention improves it, and both make it worse.** Semi-natural's interior error rises
   from 18.8% at baseline to 25.7%, 28.9% and 24.3%. On Test B pre-training takes it from 26.0% to
   42.8%. This is the same conclusion the aggregate reached, measured on the pixels that carry the
   error.

Forest and grassland do hold low interior floors — 0.6% and 5.1% on Test A — so for those two the
error genuinely is concentrated near boundaries.

---

## The original two-seed write-up, and the confound it contained

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
| Settlement | 0.713 | 0.886 | **+17.3 pp** |
| Forest | 0.718 | 0.850 | **+13.2 pp** |
| Grassland | 0.850 | 0.896 | +4.6 pp |
| **Seminatural** | 0.402 | 0.434 | **+3.2 pp** |
| Cropland | 0.350 | 0.378 | +2.8 pp |
| macro foreground | 0.607 | 0.689 | +8.2 pp |

**Ten seeds, per seed, read from the JSON rather than the console table** — the console prints only
three of the five classes, which is what led to the transcription error corrected on 2026-07-29.

## What it means

**Two classes fail at their edges. Three do not.** Settlement and forest recover 13 to 17 points once
the boundary band is removed. Grassland, cropland and semi-natural recover 3 to 5 points and stay
where they were — semi-natural still scores 0.434 with every pixel within 4 m of a boundary thrown
away.

The split is not "the weak classes versus the strong ones". It is **the classes with a physical edge
versus the classes without one.** A settlement or a forest has a real perimeter you can trace. A field
boundary between improved and semi-natural grassland is a management gradient, and cropland against
grassland is a decision about what a field currently holds. For those three the error is inside the
parcel.

That is whole-parcel misassignment: the outline was drawn correctly and the field was called the
wrong type — or the field genuinely could be either. It is not an edge-precision problem, and no
amount of more careful tracing addresses it.

## THE SEED CONTROL ON THE INTERIOR EFFECT — run 2026-07-28 (Slurm 652598)

Per class **and** per seed, which neither existing output provided. Interior means at least 32 m from
any ground-truth boundary; boundary-free tiles excluded. Rates are the mean over ten seeds of the
per-seed rate, so they are not ensemble figures.

**Test A** — 294 tiles, none boundary-free

| class | interior error, baseline | support | across-cell CV | across-seed CV |
|---|---|---|---|---|
| Seminatural | **26.98% +/- 18.78** | 1,704,635 | 34.3% | 46.8% |
| Cropland | **76.97% +/- 17.13** | 98,059 | 22.4% | 25.1% |
| Grassland | 5.15% +/- 1.70 | 14,006,571 | 29.2% | 27.4% |
| Forest | 0.75% +/- 0.21 | 475,555 | 52.4% | 58.0% |
| Settlement | *unsupported* | 2,907 | — | — |

**Test B** — 172 tiles after excluding 19 boundary-free

| class | interior error, baseline | support | across-cell CV | across-seed CV |
|---|---|---|---|---|
| Seminatural | **38.73% +/- 17.14** | 10,090,826 | 35.8% | 47.5% |
| Forest | 23.48% +/- 10.00 | 141,174 | 37.8% | 42.3% |
| Cropland | 6.38% +/- 14.89 | 419,357 | 114.2% | 119.8% |
| Grassland | 2.28% +/- 1.90 | 1,056,820 | 76.7% | 80.8% |
| Settlement | *unsupported* | 855 | — | — |

**The verdict, and it goes both ways.**

**The level survives, and it is the finding.** Semi-natural grassland is wrong on 27% of its pixels
more than 32 m from any boundary on Test A, across 1.7 million pixels, and 39% across 10.1 million on
Test B. Cropland is wrong on 77% of its interior pixels on Test A. Neither is a boundary problem.

**The directional claim dies. Do not write it.** For every class on both test sets the spread across
the four cells is no larger than the spread across the ten seeds — in the two classes that matter it
is smaller. So "both interventions make semi-natural's interior error worse" is seed noise, exactly
as the boundary-rate version of this comparison was. The earlier 18.8 -> 25.7 / 28.9 / 24.3 sequence
came from ensemble figures with no seed spread attached and must not be quoted.

**One consequence for the wording.** The per-seed baseline is 26.98% where the ten-seed ensemble gave
18.8%. The ensemble is far better than the average member, which is expected and is why per-seed is
the honest reporting unit here.

## Independent confirmation, from adjacency rather than distance

Found 2026-07-28 by a verifier auditing a different script, so it is arrived at from a direction that
shares no arithmetic with the tables below.

The two grassland classes **barely touch each other on the ground.** They share 6,578 four-neighbour
boundary contacts, which is **1.59% of all foreground-to-foreground contacts** in the scored tiles.
And only **6.13% of semi-natural reference pixels lie within 8 m of any grassland pixel** — yet
**39.15% of semi-natural pixels are predicted grassland.**

**Therefore at least 84.3% of the semi-natural-called-grassland pixels are more than 8 m from the
nearest grassland pixel.** They are not near the other class at all.

The single largest error mode in this study is **bulk regional misclassification** — whole areas of
semi-natural ground called grassland — and not confusion across a shared edge. That is the same
conclusion as the distance tables, reached without using distance-to-any-boundary at all.

**It also constrains how the pair ratio may be described.** The 2.10x reported for this pair is a
CO-AREA null: it asks whether the pair fails more than its share of the scene predicts. An adjacency
null — how much the two classes actually touch — gives **29.3x**. The co-area figure is the
conservative of the two and is the one to report, but it must be called a co-area null and not
"what area predicts", because area is not the mechanism.

## The stronger evidence: error rate against distance, per class

The trimap gain is one summary of a curve. The curve itself is in the same JSON, thirteen distance
bins per class. **Both tables in this section are the ten-seed values from
`analysis/label_ceiling/test/boundary_trimap_stage1_baseline.json`.** They were transcribed from a
two-seed probe until 2026-07-29 and two rows had been swapped in the process — the figures given for
Forest were Settlement's. Support counts are dropped rather than re-transcribed; they are in the
artifact.

| class | 0-0.5 m | 1-2 m | 4-6 m | 8-12 m | 16-24 m | 32 m+ |
|---|---|---|---|---|---|---|
| Forest | 41.7% | 22.2% | 5.6% | 4.6% | 7.4% | **0.6%** |
| Settlement | 39.1% | 18.5% | 4.3% | 5.1% | 2.4% | **0.0%** |
| Grassland | 51.7% | 24.7% | 7.9% | 6.2% | 5.6% | **5.1%** |
| Seminatural | 66.8% | 61.5% | 48.6% | 40.0% | 35.0% | **18.8%** |
| Cropland | 81.3% | 71.6% | 57.6% | 51.9% | 46.3% | **78.8%** |

**Read the last column.** Forest falls to 0.6% and settlement to 0.0% — those two really do collapse
to a near-zero interior. Grassland holds a 5.1% floor across fourteen million pixels. Semi-natural
holds **18.8%** across 1.7 million. (This paragraph said 7.1% and 13.5% until 2026-07-29, when it was
corrected against the table above it; those were transcribed from a two-seed probe.) Cropland has no boundary structure at all: 81% at the edge, 46%
in the middle distance, 79% again deep inside.

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
