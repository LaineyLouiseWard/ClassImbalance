#!/usr/bin/env python3
"""Is the 8 m band carrying the adjacency result?

The band is arbitrary. `docs/audit/PREREGISTRATION_P1_AMENDMENT.md:158` says so -- "the 8 m band
predates this amendment and is frozen. It is not neutral" -- and `docs/DO_NOT_ADD.md` rules out
citing Kohli for it, because his "8" is 8 PIXELS on 320x213 images and ours is 8 metres. The only
defence of the width itself is that it was fixed before any model was trained.

So this measures whether the width matters. For each band it reports the share of grassland's
ground lying within that distance of any semi-natural, and of any forest, over the 90 Test A
scoring chips. Reference masks only; no model output enters it. If the ratio between the two holds
across the sweep, the choice of 8 m is not carrying the claim and a reviewer asking about it can be
answered with one sentence.

Run:
    PYTHONPATH=. python scripts/analysis/adjacency_band_sweep.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

SPLIT_TAG = os.environ.get("SPLIT_TAG", "f1")
BANDS = (2.0, 4.0, 8.0, 16.0, 32.0)
GRASS, SEMI, FOREST = 2, 5, 1


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "artifacts").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def main() -> int:
    root = find_repo_root()
    import sys
    sys.path.insert(0, str(root))
    from geoseg.geo import gsd_for

    keep = set(json.loads((root / f"artifacts/scoring_subset_{SPLIT_TAG}.json").read_text())
               ["splits"]["test"]["tiles"])
    mask_dir = root / f"data/split_{SPLIT_TAG}/test/masks"
    tiles = sorted(p for p in mask_dir.glob("*.png") if p.stem in keep)
    if len(tiles) != len(keep):
        raise SystemExit(f"expected {len(keep)} subset chips in {mask_dir}, found {len(tiles)}")

    near = {b: {"semi": 0, "forest": 0} for b in BANDS}
    grass_px = 0
    for p in tiles:
        m = np.array(Image.open(p).convert("L"))
        g = m == GRASS
        if not g.any():
            continue
        grass_px += int(g.sum())
        sampling = gsd_for(p.stem)
        for cls, key in ((SEMI, "semi"), (FOREST, "forest")):
            if not (m == cls).any():
                continue
            d = ndimage.distance_transform_edt(m != cls, sampling=sampling)
            for b in BANDS:
                near[b][key] += int((g & (d < b)).sum())

    out = {"split": "test", "n_chips": len(tiles), "grassland_px": grass_px, "bands": {}}
    for b in BANDS:
        s = 100 * near[b]["semi"] / grass_px
        f = 100 * near[b]["forest"] / grass_px
        out["bands"][f"{b:g}"] = {"within_semi_pct": s, "within_forest_pct": f,
                                  "forest_over_semi": f / s}
    p = root / "artifacts/adjacency_band_sweep.json"
    p.write_text(json.dumps(out, indent=2))

    print(f"{len(tiles)} chips, {grass_px:,} grassland px")
    print(f"{'band':>7}{'semi-nat':>11}{'forest':>10}{'ratio':>9}")
    for b in BANDS:
        r = out["bands"][f"{b:g}"]
        print(f"{b:6.0f}m{r['within_semi_pct']:10.2f}%{r['within_forest_pct']:9.2f}%"
              f"{r['forest_over_semi']:8.1f}x")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
