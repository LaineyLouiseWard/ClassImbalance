#!/usr/bin/env python3
"""Ground area of the non-overlapping scoring subsets, covered vs labelled.

Test A (inland) is scored on a 90-chip non-overlapping subset, Test B (upland) on 51 chips
(`artifacts/scoring_subset_f1.json`). Every chip is 512x512 px; the inland GSD is 0.5 m, so
0.25 m2/px, and the covered area of the inland subset follows directly. The labelled FRACTION
(foreground pixels over all pixels) is pixel-size independent, so it is reported for both
subsets without the anisotropic upland pixel sizes. Background (class 0) is ground outside the
annotated farm extent, i.e. unlabelled.

Writes artifacts/scored_area_f1.json; reproduces the km2 and labelled-fraction figures in
Methods. Reads gitignored masks under data/split_f1/; a fresh clone reads the committed JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "artifacts").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("repo root not found")


ROOT = find_repo_root()
INLAND_PX_M2 = 0.25  # 0.5 m GSD inland, both axes


def class_counts(split_dir: str, tile_ids: list[str]) -> np.ndarray:
    tot = np.zeros(256, dtype=np.int64)
    for tid in tile_ids:
        m = np.asarray(Image.open(ROOT / f"data/split_f1/{split_dir}/masks/{tid}.png"))
        tot += np.bincount(m.ravel(), minlength=256)
    return tot


def main() -> None:
    sub = json.loads((ROOT / "artifacts/scoring_subset_f1.json").read_text())
    splits = sub["splits"]
    out: dict[str, dict] = {}
    for split_dir in ("test", "external_test"):
        ids = splits[split_dir]["tiles"]
        tot = class_counts(split_dir, ids)
        all_px = int(tot.sum())
        labelled_px = int(tot[1:].sum())
        rec = {
            "n_chips": len(ids),
            "all_px": all_px,
            "labelled_px": labelled_px,
            "labelled_frac_pct": 100.0 * labelled_px / all_px,
        }
        if split_dir == "test":  # inland only: isotropic 0.25 m2/px
            rec["covered_km2"] = all_px * INLAND_PX_M2 / 1e6
            rec["labelled_km2"] = labelled_px * INLAND_PX_M2 / 1e6
        out[split_dir] = rec

    p = ROOT / "artifacts/scored_area_f1.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {p}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
