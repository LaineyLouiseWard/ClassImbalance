"""Shared helpers for reproducible robustness analysis scripts."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

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
    path = VAL_ROOT / stage_dir / "confusion_matrix.csv"
    with open(path, newline="") as f:
        reader = csv.reader(f)
        return [[int(x) for x in row] for row in reader]


def load_metrics(json_path: Path) -> dict:
    """Load a metrics.json file."""
    with open(json_path) as f:
        return json.load(f)


def load_val_metrics(stage_dir: str) -> dict:
    """Load val metrics.json for a given stage directory."""
    return load_metrics(VAL_ROOT / stage_dir / "metrics.json")


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