#!/usr/bin/env python
"""
Compute ONE percentile stretch per site, so the same ground is never scaled two different ways.

WHY. The inherited reader (`_normalize_percentile`, from ODOS's starter code) computed 2-98
percentiles PER TILE. Tiles overlap 50%, so a single field enters the network under up to four
different stretches. Measured on this dataset: raw values identical on shared ground, but the
normalised values differ on 91-99.8% of shared pixels, mean 8-59 DN, max 88 of 255. Consequence,
measured on real softmax dumps over 69 overlapping pairs: 2.20% of shared foreground pixels change
class depending on which tile they were predicted from, and that instability is 29x higher within
0.5 m of a label boundary (15.21%) than in field interiors (0.52%).

That matters because the paper's claim is that residual error concentrates at label boundaries
because the labels are ambiguous there. A model is near its decision threshold at a boundary and
confident in a field interior, so ANY input perturbation flips predictions at boundaries and
nowhere else -- producing the same signature from an entirely different cause. Removing the
per-tile variation removes that rival explanation. See docs/CORRECTIONS_PAPER_PT2.md.

WHICH TILES EACH SITE'S STATS COME FROM, and why they differ:
  biodiversity (inland) -- TRAINING tiles only. Val and test are the same scene, so using their
      pixels would put held-out statistics into the training pipeline.
  ireland1, ireland2 (uplands) -- their own tiles, because both sites are held out whole and have
      no training tiles at all. This uses no labels and no predictions: it is the same thing an
      operator does when stretching a newly acquired scene. It must be stated in the methods.

OpenEarthMap is not covered here. Those tiles are already uint8 0-255 and need no stretch; the
inherited reader was re-stretching them, which this change also stops.

Output: artifacts/normalisation_stats_<SPLIT_TAG>.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio

SPLIT_TAG = os.environ.get("SPLIT_TAG", "f1")
PX_PER_TILE = 20_000       # random pixels sampled per tile per band
SEED = 42


def find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for _ in range(10):
        if (p / "data").is_dir() and (p / "scripts").is_dir():
            return p
        if p == p.parent:
            break
        p = p.parent
    return Path(__file__).resolve().parents[2]


def site_of(tile_id: str) -> str:
    return str(tile_id).split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=f"artifacts/spatial_split_manifest_{SPLIT_TAG}.json")
    ap.add_argument("--out", default=f"artifacts/normalisation_stats_{SPLIT_TAG}.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = find_repo_root()
    out = root / args.out
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to regenerate")

    # This reads the RAW imagery, which is deliberately not staged to the cluster
    # (docs/audit/BRIEF_STAGE_TO_SONIC.md: "data/biodiversity_raw (9 GB) is not needed"). The output
    # is committed, so this script should never need to run there -- but if a rebuilt split ever
    # triggers the RUNBOOK B4 fallback on Sonic, say why rather than failing per-tile.
    raw = root / "data/biodiversity_raw/images"
    if not raw.is_dir():
        raise SystemExit(
            f"{raw} not found. This script needs the raw imagery, which is not staged to the "
            "cluster. Build it where the raw data lives, commit the JSON, and pull it there.")

    man = json.loads((root / args.manifest).read_text())
    assign = man["assignment"]

    # Inland: training tiles only. Uplands: everything they have (they are held out whole, so
    # there is nothing else to use). Both rules are recorded in the artefact.
    use = defaultdict(list)
    for tid, split in sorted(assign.items()):
        s = site_of(tid)
        if s == "biodiversity":
            if split == "train":
                use[s].append(tid)
        else:
            use[s].append(tid)

    rng = np.random.default_rng(SEED)
    sites = {}
    for s, tids in sorted(use.items()):
        samples = None
        for tid in tids:
            with rasterio.open(root / "data/biodiversity_raw/images" / f"{tid}.tif") as src:
                a = src.read().astype(np.float32)          # (C,H,W)
            if samples is None:
                samples = [[] for _ in range(a.shape[0])]
            for b in range(a.shape[0]):
                # Same exclusions as the inherited reader: NaN and exact zeros are nodata here.
                v = a[b].ravel()
                v = v[~np.isnan(v)]
                v = v[v != 0]
                if v.size == 0:
                    continue
                take = min(PX_PER_TILE, v.size)
                samples[b].append(rng.choice(v, size=take, replace=False))
        p2, p98, n = [], [], []
        for b in range(len(samples)):
            v = np.concatenate(samples[b]) if samples[b] else np.array([0.0, 1.0], np.float32)
            lo, hi = np.percentile(v, (2, 98))
            # A degenerate band would make the reader divide by zero; the inherited code silently
            # passed the raw band through instead. Fail here rather than reproduce that.
            if not hi > lo:
                raise SystemExit(f"{s} band {b}: p98 ({hi}) not greater than p2 ({lo})")
            p2.append(float(lo)); p98.append(float(hi)); n.append(int(v.size))
        sites[s] = {"p2": p2, "p98": p98, "n_pixels_sampled": n, "n_tiles": len(tids),
                    "tiles_used": ("train only" if s == "biodiversity"
                                   else "all tiles of this held-out site (no training tiles exist)")}
        print(f"{s}: {len(tids)} tiles, bands={len(p2)}")
        for b in range(len(p2)):
            print(f"   band {b}: p2={p2[b]:.6f}  p98={p98[b]:.6f}  (from {n[b]:,} px)")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"split_tag": SPLIT_TAG, "manifest": args.manifest, "seed": SEED,
         "px_per_tile_sampled": PX_PER_TILE,
         "excludes": "NaN and exact zeros, matching the inherited per-tile reader",
         "note": "One stretch per site, replacing the inherited per-tile 2-98 percentile stretch. "
                 "OpenEarthMap is not listed: those tiles are already uint8 0-255 and are no "
                 "longer stretched at all.",
         "sites": sites}, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
