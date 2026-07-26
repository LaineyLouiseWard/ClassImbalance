"""Shared helpers for reproducible robustness analysis scripts."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

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

def spatial_blocks(split_root: Path, split: str, block_m: float = 950.0) -> dict:
    """{tile_id: block_id} on a grid of `block_m` cells, per CRS. THE BOOTSTRAP UNIT.

    Tiles are chipped on a 50% stride, so neighbours repeat ground and resampling tile ids treats
    dependent tiles as independent draws: 294 test tiles are only ~105 pixel-disjoint footprints, and
    ~14 independent units at the 950 m autocorrelation scale. Intervals built on tile ids are roughly
    1.6x too narrow at footprint level and 10-26x too narrow at the correlation scale (round-2 audit,
    item B6).

    WHY A GRID AND NOT CONNECTED COMPONENTS. The obvious reading of "non-overlapping footprint group"
    is a single-linkage merge of overlapping tiles. That is wrong here and was tried first: a
    contiguous test strip is one connected component under overlap (A overlaps B, B overlaps C), so
    all 294 tiles collapse into ONE group and the bootstrap becomes degenerate. A grid partitions
    space instead of chaining through it, so the unit count reflects area rather than connectivity.

    block_m defaults to 950 m, the measured correlogram range, matching the scale used for class
    support. Pass 256.0 (one tile footprint) for the pixel-disjoint count instead — that is the less
    conservative unit, and the two should be reported together.
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
        by_crs[r[4]].append(r)
    for crs, rs in by_crs.items():
        lat = float(np.mean([r[2] for r in rs]))
        if rs[0][3]:
            sx = block_m / (111320.0 * math.cos(math.radians(lat)))
            sy = block_m / 111132.0
        else:
            sx = sy = block_m
        for tid, cx, cy, _, _ in rs:
            out[tid] = (crs, int(math.floor(cx / sx)), int(math.floor(cy / sy)))
    return out


def resample_blocks(tiles, blocks: dict, rng):
    """One bootstrap draw: resample BLOCKS with replacement, return the tiles they contain."""
    from collections import defaultdict
    by_block = defaultdict(list)
    for t in tiles:
        by_block[blocks.get(t, t)].append(t)
    keys = sorted(by_block)
    pick = rng.choice(len(keys), len(keys), replace=True)
    return [t for k in pick for t in by_block[keys[k]]], len(keys)
