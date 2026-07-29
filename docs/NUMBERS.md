# Where the paper's numbers come from

**Generated — do not hand-edit.** Rebuild with:

    PYTHONPATH=. python scripts/analysis/verify_narrative_numbers.py --markdown

Every number the write-up quotes has a row here, the committed artifact that holds it, and
the command that regenerates that artifact. To check them all at once, run the same script
with no arguments; it exits non-zero if any has drifted.

77 numbers, grouped by the command that produces them.

## 1. rho and the near/interior error rates

```
PYTHONPATH=. python scripts/analysis/boundary_rate_ratio.py --split-root <dedup|split_f1> --split test --softmax-root <cluster> --cell <cell> --seeds <seed> --band-m <band>
```

| number | value |
|---|---|
| rho, dedup subset, 8 m | 2.13676 |
| rho, dedup subset, 1 m | 3.69248 |
| rho, all tiles, 8 m | 2.28227 |
| rho, all tiles, 1 m | 3.84773 |
| near-boundary rate, all tiles, 8 m, % | 19.6338 |
| near-boundary rate, all tiles, 1 m, % | 41.1749 |
| near-boundary rate CV over 40 runs, 8 m, % | 3.47057 |
| near-boundary rate CV over 40 runs, 1 m, % | 1.96407 |

## 2. Class adjacency and the contact null

```
PYTHONPATH=. python scripts/analysis/class_adjacency.py --split test --confusion analysis/confusion/confusion_test_stage1_baseline.npy
```

| number | value |
|---|---|
| S->G bound: share not at a shared edge, % | 84.3533 |
| G->S bound: share not at a shared edge, % | 87.5564 |
| C->G bound: share not at a shared edge, % | 77.6422 |
| contact-null ratio, grassland pair | 29.2846 |
| total foreground contacts | 412641 |
| grassland pair contact share, % | 1.59412 |
| semi-natural within 8 m of grassland, % | 6.12631 |
| grassland within 8 m of semi-natural, % | 0.599259 |

## 3. Scored-subset ground area (covered vs labelled)

```
PYTHONPATH=. python scripts/analysis/compute_scored_area.py
```

| number | value |
|---|---|
| Test A subset covered, km2 | 5.89824 |
| Test A subset labelled, % | 88.2029 |
| Test B subset labelled, % | 55.7509 |

## 4. Interior error rate by class

```
PYTHONPATH=. python scripts/analysis/interior_error_by_class.py --softmax-root <cluster> --split test --cell <cell> --out-dir analysis/interior
```

| number | value |
|---|---|
| interior error, semi-natural, Test A, % | 26.9841 |
| interior error, cropland, Test A, % | 76.9655 |
| interior error, grassland, Test A, % | 5.14846 |
| interior error, forest, Test A, % | 0.749819 |
| interior error, semi-natural, Test B, % | 38.7338 |
| interior support, semi-natural, Test A, px | 1.70464e+06 |
| interior support, settlement, Test A, px | 2907 |
| interior error, forest, Test B, % | 23.4777 |
| interior error, grassland, Test B, % | 2.28069 |
| interior error, cropland, Test B, % | 6.37581 |
| interior support, semi-natural, Test B, px | 1.00908e+07 |
| interior support, settlement, Test B, px | 855 |

## 5. Class-pair ratios, confusion summary, sampler effect, per-class contrasts

```
PYTHONPATH=. python scripts/analysis/narrative_numbers.py --confusion-dir analysis/confusion --metrics-dir analysis/metrics/test --split test --out artifacts/narrative_numbers_test.json
```

| number | value |
|---|---|
| pair share of foreground error, % | 46.6832 |
| pair co-area ratio | 2.10232 |
| pair error pixels | 1.28993e+07 |
| semi-natural predicted vs reference area, % | 11.4202 |
| settlement predicted vs reference area, % | 5.10695 |
| cropland predicted vs reference area, % | -45.6245 |
| semi-natural recall, % | 56.0271 |
| semi-natural precision, % | 50.2845 |
| semi-natural called grassland, % | 39.1539 |
| grassland called semi-natural, % | 4.81578 |
| sampler: extra predicted semi-natural px | 1.78543e+06 |
| sampler: share of extra that was correct, % | 3.04442 |
| sampler: semi-natural recall after, % | 56.4002 |
| OEM pre-training, mIoU, pp | -0.366965 |
| sampler, mIoU, pp | 0.19019 |
| OEM pre-training, mIoU, CI low | -3.67361 |
| OEM pre-training, mIoU, CI high | 2.93968 |
| sampler, mIoU, CI low | -1.97905 |
| sampler, mIoU, CI high | 2.35943 |
| interaction, mIoU, pp | -2.08178 |
| interaction, Cropland IoU, pp | -10.4437 |
| interaction, mIoU, CI low | -3.81014 |
| interaction, mIoU, CI high | -0.353427 |

## 6. The seed-only control

```
PYTHONPATH=. python scripts/analysis/narrative_numbers.py --rho-dir analysis/rho_dedup --split test --band-m <band> --out artifacts/seedcontrol_dedup_band<band>.json
```

| number | value |
|---|---|
| seed control 8 m: near CV across cells, % | 3.76081 |
| seed control 8 m: near CV across seeds, % | 3.69598 |
| seed control 8 m: interior CV across cells, % | 13.886 |
| seed control 8 m: interior CV across seeds, % | 14.2061 |

## 7. Pair error geometry: distance and component size

```
PYTHONPATH=. python scripts/analysis/pair_error_geometry.py --softmax-root <cluster> --split test --cell <cell> --out-dir analysis/pair_geometry
```

| number | value |
|---|---|
| S->G guarded beyond 8 m, % | 68.5202 |
| G->S guarded beyond 8 m, % | 75.3287 |
| F->G guarded beyond 8 m, % | 4.83165 |
| S->G mass in components >1 ha, % | 56.1692 |
| G->S mass in components >1 ha, % | 53.8345 |
| F->G mass in components >0.1 ha, % | 10.4438 |
| F->G median component, m2 | 1.025 |
| S->G beyond 8 m, unguarded, % | 68.3955 |
| G->S beyond 8 m, unguarded, % | 76.2125 |
| S->G mass in components >0.1 ha, % | 90.6248 |
| G->S mass in components >0.1 ha, % | 92.9218 |
| S->G error px per seed | 570425 |
| G->S error px per seed | 719500 |

## 8. Pooled confusion matrices

```
PYTHONPATH=. python scripts/analysis/pooled_confusion.py --softmax-root <cluster> --split test --cell <cell> --out-dir analysis/confusion
```

| number | value |
|---|---|
| foreground errors, Test A baseline | 2.76314e+07 |
| foreground pixels scored (10 seeds) | 2.08097e+08 |
| grassland share of Test A foreground, % | 71.7958 |

## 9. Class support

```
PYTHONPATH=. python scripts/analysis/report_class_support.py
```

| number | value |
|---|---|
| train grassland pixels | 1.9367e+08 |
| train semi-natural pixels | 1.17807e+07 |
| Test A foreground, all 294 tiles | 6.95587e+07 |

## Not in this ledger, and why

- **Test B** is covered only where the narrative quotes it. If more Test B figures enter
  the paper, they need rows.
- **Per-tile illustrations** (individual parcels in the qualitative panel) come from softmax
  dumps that are not committed, being ~50 MB. They are illustrative; the component
  statistics carry the claim and those are here.
