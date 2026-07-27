"""Shared helpers for reproducible robustness analysis scripts."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

# Metres per degree: ONE definition, in geoseg/geo.py. These two literals appeared in seven
# files with no derivation, and terrain_separability used the LONGITUDE constant for the
# latitude direction.
from geoseg.geo import M_PER_DEG_LAT, M_PER_DEG_LON_EQ  # noqa: E402,F401

# ── Repo root detection ─────────────────────────────────────────────────────

def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__)).resolve()
    for p in [start, *start.parents]:
        if (p / "artifacts").is_dir() and (p / "geoseg").is_dir():
            return p
    raise FileNotFoundError(f"Could not find repo root from {start}")


REPO_ROOT = find_repo_root()

# ── Split-scoped artefacts ──────────────────────────────────────────────────

# Which fold's artefacts these scripts analyse. RUNBOOK.sh exports SPLIT_TAG; f1 is its default too.
SPLIT_TAG = os.environ.get("SPLIT_TAG", "f1")


def cell_dir(folder: str) -> str:
    """The evaluation_results/ directory a cell's metrics land in, for the current SPLIT_TAG.

    compute_metrics.py names its output directory after the CHECKPOINT'S PARENT DIRECTORY
    (`safe_name = ckpt.parent.name`), and every campaign checkpoint lives in a tagged directory,
    so it writes `<cell>_<tag>/`. A reader that spells the cell untagged does not fail: the
    withdrawn campaign's untagged directories are still on disk under
    evaluation/evaluation_results/val/, so it silently resolves to leaked numbers and exits 0.
    Every reader goes through here so the writer and the readers cannot drift apart again.
    """
    return f"{folder}_{SPLIT_TAG}"


def assert_metrics_provenance(m: dict, path, folder: str, expect_data_root_name: str,
                              expect_tta: bool = False) -> None:
    """Reject a metrics.json that did not come from this cell, this split tag AND this split.

    ONE implementation, imported by every reader, because the previous version of this check was
    copy-pasted into three places and therefore had the same blind spot three times: it validated
    `checkpoint` and `tta` and nothing else. That is enough to catch a different CELL and a
    different SPLIT_TAG, and not enough to catch a different SPLIT or a different DATASET.

    Demonstrated 2026-07-26: taking the withdrawn campaign's metrics and changing ONLY the
    checkpoint basename was sufficient to make `export_final_test_table.py` build the manuscript's
    headline test table and exit 0, from files whose own fields read
    `split: val`, `data_root: .../biodiversity_split/val` -- the withdrawn LEAKING split, and the
    validation split at that, presented as held-out test. Both fields were sitting in the file,
    unread. So read them.

    `expect_data_root_name` is the split directory the metrics must have been computed over
    ("val", "test", "external_test"). It is the discriminator that `split` cannot be, because
    compute_metrics.py takes `--split test` for BOTH Test A and Test B and tells them apart only
    by `--data-root`.
    """
    p = Path(path)
    got_ckpt = Path(m.get("checkpoint", "")).name
    want_ckpt = f"{cell_dir(folder)}.ckpt"
    if got_ckpt != want_ckpt:
        raise SystemExit(
            f"{p}\n  was produced by checkpoint '{got_ckpt or '(none recorded)'}', not "
            f"'{want_ckpt}'. This is a metrics file from another cell, another split tag, or the "
            f"withdrawn pre-2026-07-26 campaign.")

    data_root = m.get("data_root")
    if not data_root:
        raise SystemExit(
            f"{p}\n  records no data_root, so the data it was computed over cannot be established. "
            f"Re-run evaluation/compute_metrics.py; it has written this field since 2026-06.")
    dr = Path(data_root)
    if dr.name != expect_data_root_name:
        raise SystemExit(
            f"{p}\n  was computed over '{dr}', whose split directory is '{dr.name}', not "
            f"'{expect_data_root_name}'. A metrics file for one split cannot stand in for another.")
    want_parent = f"split_{SPLIT_TAG}"
    if dr.parent.name != want_parent:
        raise SystemExit(
            f"{p}\n  was computed over '{dr}', which sits under '{dr.parent.name}', not "
            f"'{want_parent}'. This is another split entirely — most likely the withdrawn "
            f"'biodiversity_split', whose held-out tiles leak into training.")

    if bool(m.get("tta")) is not bool(expect_tta):
        raise SystemExit(f"{p}\n  has tta={m.get('tta')}; this reader expects tta={expect_tta}.")


def resolve_artifact(env_var: str, template: str) -> Path:
    """Resolve a per-fold artefact path: $env_var if set, else the tagged file for $SPLIT_TAG.

    The untagged artefacts from the pre-2026-07-25 random split are still in the repo, and were
    previously the hard-coded default here. They were built on a split whose held-out tiles overlap
    the current training set, so loading one silently mixes leaked ground into an analysis. They are
    deliberately NOT a fallback: to analyse them, name them explicitly through the environment.
    """
    env = os.environ.get(env_var)
    path = REPO_ROOT / env if env else REPO_ROOT / "artifacts" / template.format(tag=SPLIT_TAG)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.relative_to(REPO_ROOT)} not found (SPLIT_TAG={SPLIT_TAG}). Build it with "
            f"RUNBOOK.sh, or set ${env_var} to the artefact you mean.")
    return path

# ── Canonical stage definitions ─────────────────────────────────────────────

# Paper ablation stages mapped to val evaluation-result folder paths (3-stage, no-replication).
# Stage 2a (OEM pre-train on the combined set) is omitted: it trains on OEM, not biodiversity,
# so only the Biodiversity-finetuned endpoint (Stage 2b) is reported in the ablation table.
STAGES = [
    ("1", "stage1_baseline"),
    ("2", "stage2b_oem_finetune"),
    ("3", "stage3_clsbal"),
]

VAL_ROOT = REPO_ROOT / "evaluation" / "evaluation_results" / "val"

# ── Class indices (0-indexed, matching confusion matrix rows/cols) ──────────

CLASS_NAMES = ["Background", "Forest", "Grassland", "Cropland", "Settlement", "Seminatural"]
IDX_BACKGROUND  = 0
IDX_FOREST      = 1
IDX_GRASSLAND   = 2
IDX_CROPLAND    = 3
IDX_SETTLEMENT  = 4
IDX_SEMINATURAL = 5

MAJORITY_INDICES = [IDX_FOREST, IDX_GRASSLAND, IDX_CROPLAND]
MINORITY_INDICES = [IDX_SETTLEMENT, IDX_SEMINATURAL]

# ── Loaders ─────────────────────────────────────────────────────────────────

def load_confusion_matrix(stage_dir: str) -> list[list[int]]:
    """Load a 6x6 confusion matrix from CSV (raw pixel counts)."""
    path = VAL_ROOT / cell_dir(stage_dir) / "confusion_matrix.csv"
    with open(path, newline="") as f:
        reader = csv.reader(f)
        return [[int(x) for x in row] for row in reader]


def load_metrics(json_path: Path) -> dict:
    """Load a metrics.json file."""
    with open(json_path) as f:
        return json.load(f)


def load_val_metrics(stage_dir: str) -> dict:
    """Load val metrics.json for a given stage directory."""
    return load_metrics(VAL_ROOT / cell_dir(stage_dir) / "metrics.json")


def load_weights_tsv(path: Path | None = None) -> dict[str, float]:
    """Load a sampler-weights TSV → {img_id: weight} (default: this fold's clsbal weights)."""
    path = path or resolve_artifact("SAMPLER_TSV", "sampler_weights_clsbal_{tag}.tsv")
    weights = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_id, w = line.split("\t")
            weights[img_id] = float(w)
    return weights


def load_augmentation_list(path: Path | None = None) -> dict:
    """Load this fold's minority-rich tile list."""
    path = path or resolve_artifact("AUG_LIST", "train_augmentation_list_{tag}.json")
    with open(path) as f:
        return json.load(f)

# ── Unit of analysis: independent ground, not tiles ─────────────────────────

# THE GRID ANCHOR, stated rather than implied. Cell membership is floor((coord - origin) / cell), and
# until 2026-07-26 the origin was simply whatever the coordinate system happened to use: the UTM 29N
# false easting of 500,000 m for the inland site, the prime meridian and the equator for the uplands.
# Two unrelated arbitrary origins, neither chosen, neither written down. Neither has anything to do
# with this landscape, so the shipped partition is ONE MEMBER of a family indexed by this offset --
# scripts/analysis/block_phase_sweep.py reports the family, and the count moves by up to 40% across
# it. (0.0, 0.0) reproduces the historical behaviour exactly; it is now a decision rather than a side
# effect of the projection. Changing it changes the shipped partition -- do not, without a dated note.
BLOCK_ORIGIN = (0.0, 0.0)


def spatial_blocks(split_root: Path, split: str, block_m: float = 950.0) -> dict:
    """{tile_id: cell_id} on a grid of `block_m` cells, per (CRS, site). A DESCRIPTIVE UNIT.

    WHAT THIS COUNTS, AND WHAT IT DOES NOT. It counts CELLS TOUCHED: how many grid cells the tiles of
    a split fall into. That is a spread statistic — how scattered the ground carrying a class is. It
    is NOT a count of independent parcels, and it was described as one until 2026-07-26. Test A's 294
    tiles touch 16 cells while covering 6.783 km2, which is 7.52 cells' worth of ground; Test B's 191
    tiles touch 14 while covering 5.164 km2, or 5.72 cells' worth. Sixteen disjoint 950 m parcels
    cannot exist inside 7.52 parcels of ground. Never write "independent 950 m blocks".

    THIS IS NOT AN INFERENTIAL UNIT AND THERE IS NO LONGER A BOOTSTRAP. `resample_blocks` was removed
    with the block bootstrap on 2026-07-26: it existed to supply the lower bound the withdrawn
    rho >= 4.0 threshold was judged on (D18), both test sets are complete enumerations of their
    ground, and uncertainty in this study is per-seed and paired (aggregate_seeds.py). See METHODS §7.

    WHY A GRID AND NOT CONNECTED COMPONENTS. The obvious reading of "non-overlapping footprint group"
    is a single-linkage merge of overlapping tiles. That is wrong here and was tried first: a
    contiguous strip is one connected component under overlap (A overlaps B, B overlaps C), so all
    294 tiles collapse into ONE group. A grid partitions space instead of chaining through it, so the
    count reflects area rather than connectivity.

    block_m defaults to 950 m, matching the scale used for class support. That is NOT the inland
    site's measured range, which this docstring claimed until 2026-07-26. Measured and committed
    under artifacts/correlogram/: inland composition 750 m (900 of 1,952 tiles), inland spectral
    1,350 m; 950 m is ireland2's composition range. Pass 256.0 (one tile footprint) for the
    pixel-disjoint count instead.
    """
    import math
    from collections import defaultdict
    import rasterio

    rows = []
    for f in sorted((split_root / split / "images").glob("*.tif")):
        with rasterio.open(f) as d:
            b = d.bounds
            rows.append((f.stem, (b.left + b.right) / 2, (b.bottom + b.top) / 2,
                         bool(d.crs and d.crs.is_geographic), str(d.crs)))
    out = {}
    by_crs = defaultdict(list)
    for r in rows:
        # Group by (CRS, SITE), not CRS alone. ireland1 and ireland2 share EPSG:4326 but sit ~50 km
        # apart at 51.54 and 52.03 degrees, and this took ONE mean latitude across both to convert
        # 950 m into degrees of longitude. That put the cell edges in a different place from
        # build_spatial_split.support_blocks, which scales per site, so the two functions disagreed
        # on how many independent blocks Test B has -- 12 against 14 -- while both claimed to be
        # counting the same thing. Per-site scaling is the correct one: a cell is 950 m at the
        # latitude where it actually lies.
        by_crs[(r[4], r[0].split("_")[0])].append(r)
    for (crs, _site), rs in by_crs.items():
        lat = float(np.mean([r[2] for r in rs]))
        if rs[0][3]:
            sx = block_m / (M_PER_DEG_LON_EQ * math.cos(math.radians(lat)))
            sy = block_m / M_PER_DEG_LAT
        else:
            sx = sy = block_m
        for tid, cx, cy, _, _ in rs:
            out[tid] = (crs, _site,
                        int(math.floor((cx - BLOCK_ORIGIN[0]) / sx)),
                        int(math.floor((cy - BLOCK_ORIGIN[1]) / sy)))
    return out


# resample_blocks was REMOVED 2026-07-26 with the block bootstrap. It drew spatial blocks with
# replacement for a percentile CI, and that CI existed to supply the lower bound the withdrawn
# rho >= 4.0 threshold was judged on (D18). Both test sets are complete enumerations of their
# ground, so there is nothing to resample; uncertainty here is per-seed and paired
# (aggregate_seeds.py). spatial_blocks stays, as DESCRIPTION only -- how many 950 m cells of ground
# the tiles touch, which is a spread statistic and not a count of independent parcels.
# Do not reintroduce a second uncertainty channel.
