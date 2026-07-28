#!/usr/bin/env python3
"""
Compute and freeze the boundary-band denominators the §P1 pre-registration depends on.

WHY THIS EXISTS. Version 1 of the amendment registered denominators of 38.0% and 22.2% that no script
in the repository could reproduce, and the two numbers turned out to use different band conventions:
38.0% was `distance <= 8.0` while 22.2% was `distance < 8.0`. They agree for the uplands only because
anisotropic sampling never lands exactly on 8.0, whereas the isotropic 0.5 m inland grid hits it on
every 16-pixel run. Since the pipeline that computes the numerator uses strict `<`, the mismatch
understated Test A. A registered number that cannot be regenerated is not registered, so this writes
them as a committed artefact.

TWO CONVENTIONS THAT MATTER, both settled here and not adjustable later:

  1. STRICT band membership, `distance < 8.0 m`, matching `EDGES_M[:-1] < INT_MIN_M` downstream.
  2. Tiles with NO ground-truth boundary are EXCLUDED. 19 of the 191 upland tiles are single-class,
     so `boundary_distance` returns inf everywhere and the near-boundary category is empty by
     construction: any error they hold is forced into the far bin and can only depress the rate ratio.
     They carry 16% of Test B foreground. The statistic is undefined where no boundary exists, so they
     leave both numerator and denominator rather than silently biasing them. Boundaries are detected
     within-tile only, so a homogeneous tile abutting a different class has its true interface
     invisible — which is the reason for excluding rather than counting them.

Run:
    PYTHONPATH=. python scripts/analysis/register_boundary_denominators.py --split-root data/split_f1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.analysis.seed_disagreement import boundary_distance

BAND_M = 8.0
SPLITS = ("test", "external_test")


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        # `artifacts` and `scripts`, NOT `data`: data/ is gitignored, so keying on it made every
        # one of these scripts raise "repo root not found" in a fresh clone -- including the ledger
        # a reviewer would run to check the paper's numbers.
        if (parent / "artifacts").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def site_of(tile_id: str) -> str:
    return tile_id.split("_")[0]


def measure(mask_dir: Path) -> dict:
    """Band area share over foreground, strict `< BAND_M`, excluding boundary-free tiles."""
    per_site = {}
    kept_fg = kept_band = 0
    dropped, dropped_fg = [], 0
    incl_fg = incl_band = 0

    for p in sorted(mask_dir.glob("*.png")):
        m = np.array(Image.open(p).convert("L"))
        fg = m != 0
        n_fg = int(fg.sum())
        if n_fg == 0:
            continue
        d = boundary_distance(m, p.stem)
        # A tile with no GT boundary has no finite distances at all.
        if not np.isfinite(d[fg]).any():
            dropped.append(p.stem)
            dropped_fg += n_fg
            kept_fg += n_fg          # counted only for the "if included" comparison
            continue
        n_band = int((fg & (d < BAND_M)).sum())
        incl_fg += n_fg
        incl_band += n_band
        kept_fg += n_fg
        kept_band += n_band
        s = per_site.setdefault(site_of(p.stem), {"fg": 0, "band": 0, "tiles": 0})
        s["fg"] += n_fg
        s["band"] += n_band
        s["tiles"] += 1

    return {
        "band_m": BAND_M,
        "convention": "strict distance < 8.0 m; boundary-free tiles excluded",
        "n_boundary_free_tiles": len(dropped),
        "boundary_free_tiles": dropped,
        "foreground_px_excluded": dropped_fg,
        "share_of_foreground_excluded": (dropped_fg / kept_fg) if kept_fg else 0.0,
        "registered_band_area_share": (incl_band / incl_fg) if incl_fg else float("nan"),
        # For the record: what the number would have been WITHOUT the exclusion. Version 1 used this.
        "band_area_share_if_boundary_free_included": (kept_band / kept_fg) if kept_fg else float("nan"),
        "foreground_px": incl_fg,
        "per_site": {k: {**v, "band_area_share": v["band"] / v["fg"],
                         "ceiling_1_over_p": v["fg"] / v["band"] if v["band"] else float("inf")}
                     for k, v in per_site.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", default="data/split_f1")
    ap.add_argument("--out", default="artifacts/boundary_band_denominators.json")
    args = ap.parse_args()

    root = find_repo_root()
    sr = root / args.split_root
    out = {"split_root": args.split_root, "splits": {}}

    label = {"test": "Test A (inland held-out strip)", "external_test": "Test B (upland sites)"}
    for split in SPLITS:
        d = sr / split / "masks"
        if not d.is_dir():
            continue
        r = measure(d)
        out["splits"][split] = r
        print(f"\n{label[split]}  —  {split}")
        print(f"  registered band area share  {100*r['registered_band_area_share']:.3f}%   "
              f"(ceiling 1/p = {1/r['registered_band_area_share']:.3f})")
        print(f"  foreground pixels           {r['foreground_px']:,}")
        if r["n_boundary_free_tiles"]:
            print(f"  EXCLUDED {r['n_boundary_free_tiles']} boundary-free tiles carrying "
                  f"{100*r['share_of_foreground_excluded']:.2f}% of foreground")
            print(f"    had they been included: {100*r['band_area_share_if_boundary_free_included']:.3f}%")
        for site, v in sorted(r["per_site"].items()):
            print(f"    {site:14} {v['tiles']:4d} tiles  p = {100*v['band_area_share']:6.3f}%  "
                  f"ceiling {v['ceiling_1_over_p']:.3f}")

    p = root / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out} — this is the registered artefact; regenerate, do not hand-edit")


if __name__ == "__main__":
    main()
