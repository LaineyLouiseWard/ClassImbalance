# Characterising Thematic Error in an Operational Rural Land-Cover Map

Code accompanying the manuscript *Characterising Thematic Error in an Operational Rural Land-Cover Map*
(in preparation for the journal *Remote Sensing*, special issue on Data Curation for AI).

This is a case study of one operational product: a five-class land-cover map of rural Irish farmland from
half-metre Pléiades imagery, produced by an industry partner from a single manual annotation pass that
cannot be repeated at scale. Holding the segmentation model fixed (FT-UNetFormer), two off-the-shelf
data-curation interventions — pre-training on the public OpenEarthMap dataset and minority oversampling —
are crossed in a 2×2 factorial over ten training seeds. Neither main effect separates from run-to-run
variation: the design resolves about three percentage points of foreground mean IoU (mIoU), wider than the
spread between the four configurations.

The contribution is diagnostic rather than algorithmic. Three reproducible measurements locate the error.
Nearly half of all foreground error is a single class pair, grassland and semi-natural grassland confused
with each other (2.1× the share their area alone would give), and it falls within fields rather than at
their shared edge: under 1% of grassland lies within eight metres of any semi-natural grassland, and about
half of the pair's error sits in connected patches larger than a hectare, while the map otherwise follows
the usual boundary pattern (a pixel within one metre of a class boundary is misclassified about 3.7× as
often as one further away). Because the error that carries the volume is not concentrated at boundaries,
re-tracing the outlines already drawn cannot reach it. Whether the cause is model failure, parcel-level
label error, or absorption of the minority class into the majority is left open.

Built with PyTorch 2.9, PyTorch Lightning 2.3, and Rasterio 1.4 on Python 3.11; the environment is pinned in
`environment.yaml`.

## Setup

```bash
conda env create -f environment.yaml
conda activate label-quality-ceiling
```

## Reproduce

A single script runs the whole pipeline end-to-end from raw data:

```bash
bash RUNBOOK.sh              # everything from scratch
bash RUNBOOK.sh --from B1    # resume from training
bash RUNBOOK.sh --from C1    # resume from evaluation
```

Evaluate the shipped model on Test A, the inland held-out strip. `SPLIT_TAG` gates every checkpoint
and evaluation path and must be exported; `--checkpoints` is explicit because `--base-dir` globs the
whole tree and would pick up the withdrawn campaign's untagged weights. No test-time augmentation is
used anywhere in the reported results.

```bash
SPLIT_TAG=f1 PYTHONPATH=. python evaluation/compute_metrics.py \
  --split test --data-root data/split_f1/test \
  --checkpoints model_weights/biodiversity/stage3_clsbal_f1/stage3_clsbal_f1.ckpt
```

Checkpoints are not distributed — `RUNBOOK.sh` produces them, and the reported results come from ten
seeds rather than one. The per-seed metrics and every derived artifact the write-up quotes *are*
committed, under `analysis/`, so the numbers can be checked without retraining:

```bash
PYTHONPATH=. python scripts/analysis/verify_narrative_numbers.py
```

Rebuild every paper figure:

```bash
python scripts/figures/build_all_figures.py
```

`RUNBOOK.md` is the detailed walkthrough: each stage (teacher build, student lineage, class-balanced sampler weights,
evaluation), the stage→config map, and the `--from` resume points. The A1–A6 robustness analyses live in
`scripts/analysis/`; the figure map is in [docs/FIGURES.md](docs/FIGURES.md).

## Data availability

The Biodiversity dataset is proprietary and not publicly available; it was acquired under licence from ODOS
Technologies and cannot be redistributed. OpenEarthMap is public at
[open-earth-map.org](https://open-earth-map.org). Pre-trained checkpoints are not redistributed. Users with licensed
access should place files as follows:

| Asset | Location |
|-------|----------|
| Biodiversity imagery & masks | `data/biodiversity_raw/` |
| Biodiversity train/val/test split | `data/biodiversity_split/` |
| OpenEarthMap raw tiles | `data/openearthmap_raw/` |
| OEM relabelled (6-class) | `data/openearthmap_relabelled/` |
| OEM filtered subset | `data/openearthmap_filtered/` |
| Stage checkpoints | `model_weights/biodiversity/<stage>/` |
| OEM teacher weights | `pretrain_weights/` |
| Stage 3 sampler weights (clsbal) | `artifacts/sampler_weights_clsbal_f1.tsv` |
| Pre-computed evaluation outputs | `evaluation/evaluation_results/` |

The RGB+NIR 4-channel ablation (the near-infrared null result discussed in the paper) is kept on the
`experiment/rgb-nir` branch, since the 4-channel data path is not backward-compatible with the RGB pipeline used for
the reported results.

## Acknowledgements

The FT-UNetFormer implementation derives from ODOS Technologies'
[GeoSeg-Biodiversity](https://github.com/odostech/GeoSeg-Biodiversity). The proprietary Biodiversity dataset was
provided by ODOS Technologies under licence. The underlying Pléiades satellite imagery is © CNES 2021, distribution
Airbus DS; it is proprietary and is not distributed in this repository, which shares only code and imagery-free
derived outputs.

## Licence and citation

Code is released under the MIT Licence (see `LICENSE`). The datasets and satellite imagery are not covered by this
licence and remain proprietary. Citation details will be added upon publication.
