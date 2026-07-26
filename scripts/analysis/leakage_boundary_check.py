#!/usr/bin/env python
"""
Does the boundary / label-quality-ceiling result survive the train-test tile overlap?

The Biodiversity tiles are chipped on a 50% stride (256 x 256 m footprints on a 128 m grid), and
the split is random by tile, so ~93% of the mean held-out tile's ground area also sits inside a
training tile -- same pixels, same labels. Interiors may therefore be memorised, which would push
the interior error rate artificially toward zero and inflate the boundary-vs-interior contrast the
paper rests on.

This re-stratifies every headline boundary number by TRAINING EXPOSURE, using the per-seed softmax
dumps already on disk. No retraining, no GPU.

  unseen  = pixels covered by NO training tile footprint
  seen    = pixels covered by >= 1 training tile footprint

Reported for each cell/split:
  A  error rate vs distance-to-GT-boundary, unseen vs seen        (the Figure 8b curve)
  B  per-class boundary band (<=1.5 m) vs deep interior (>8 m)     ("every class collapses")
  C  share of foreground error within 8 m of a boundary            (the "92% / 96%" claim)
  D  trimap IoU recovery on unseen pixels only                     (the "90.1 -> 98.0" claim)
  E  plain macro foreground IoU, unseen vs seen                    (the leakage inflation itself)
  F  dose-response: error and entropy by number of covering training tiles
  G  within-tile control: A restricted to tiles holding both groups

Sanity gate: on ALL pixels the script must reproduce the stored manuscript numbers in
analysis/label_ceiling/boundary_trimap_<cell>.json. If it does not, this script is wrong.

Usage:
    PYTHONPATH=. python scripts/analysis/leakage_boundary_check.py
    PYTHONPATH=. python scripts/analysis/leakage_boundary_check.py --only val:stage3_clsbal
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

import os  # noqa: E402
# The untagged split directories belong to the WITHDRAWN random split (219 val / 218 test tiles).
SPLIT_TAG = os.environ.get("SPLIT_TAG", "f1")


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.seed_disagreement import (  # noqa: E402
    STUDENT_CLASSES, C, GSD_M, DIST_BIN_EDGES_PX, DIST_BIN_EDGES_M,
    boundary_distance, load_mask, load_seed_stack, list_val_tiles, tile_uncertainty,
)
from scripts.analysis.boundary_trimap_iou import (  # noqa: E402
    RADII_M, conf_over, iou_from_conf, FOREGROUND,
)

SEEDS = list(range(42, 52))
BND_MAX_M, INT_MIN_M = 1.5, 8.0
# Exposure buckets: how many training-tile footprints cover the pixel.
EXPOSURE_BINS = [0, 1, 2, 3, 4, 5]          # 5 = "5 or more"

# Runs: (split, cell, softmax_root, mask_dir, reference json for the sanity gate)
RUNS = [
    ("val", "stage3_clsbal", "sonic/results", f"data/split_{SPLIT_TAG}/val/masks",
     "analysis/label_ceiling/boundary_trimap_stage3_clsbal.json"),
    ("val", "stage1_baseline", "sonic/results", f"data/split_{SPLIT_TAG}/val/masks",
     "analysis/label_ceiling/boundary_trimap_stage1_baseline.json"),
    ("test", "stage3_clsbal", "analysis/test_softmax", f"data/split_{SPLIT_TAG}/test/masks",
     "analysis/label_ceiling_test/boundary_trimap_stage3_clsbal.json"),
]


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "data").is_dir() and (parent / "analysis").is_dir():
            return parent
    raise RuntimeError("repo root not found")


# ---------------------------------------------------------------------------
# Step 1: training-exposure maps
# ---------------------------------------------------------------------------
def crs_group(tile_id: str) -> str:
    """Tiles come in two coordinate systems; footprints are only comparable within a group."""
    return tile_id.split("_")[0]


def read_bounds(root: Path, cache: Path | None = None) -> dict:
    """{tile_id: (split, left, bottom, right, top)} for every tile in the split directories.

    Opening all 2,143 GeoTIFFs takes a couple of minutes, so the result is cached.
    """
    if cache is not None and cache.exists():
        return {k: tuple(v) for k, v in json.loads(cache.read_text()).items()}
    out = {}
    for split in ("train", "val", "test"):
        for f in sorted((root / "data/biodiversity_split" / split / "images").glob("*.tif")):
            with rasterio.open(f) as d:
                b = d.bounds
                out[f.stem] = (split, b.left, b.bottom, b.right, b.top)
    if cache is not None:
        cache.write_text(json.dumps(out))
    return out


def exposure_map(tile_id: str, bounds: dict, train_by_group: dict, shape) -> np.ndarray:
    """Per-pixel count of training-tile footprints covering that pixel (uint8, saturating)."""
    _, L, B, R, T = bounds[tile_id]
    H, W = shape
    cover = np.zeros((H, W), dtype=np.uint8)
    for jid in train_by_group.get(crs_group(tile_id), ()):
        _, jl, jb, jr, jt = bounds[jid]
        if jr <= L or jl >= R or jt <= B or jb >= T:
            continue
        # raster row 0 is the north edge
        c0 = int(round((max(L, jl) - L) / (R - L) * W))
        c1 = int(round((min(R, jr) - L) / (R - L) * W))
        r0 = int(round((T - min(T, jt)) / (T - B) * H))
        r1 = int(round((T - max(B, jb)) / (T - B) * H))
        c0, c1 = max(0, c0), min(W, c1)
        r0, r1 = max(0, r0), min(H, r1)
        if c1 > c0 and r1 > r0:
            cover[r0:r1, c0:c1] += 1
    return cover


# ---------------------------------------------------------------------------
# Accumulators
# ---------------------------------------------------------------------------
NB = len(DIST_BIN_EDGES_M) - 1
EDGES_M = DIST_BIN_EDGES_M
BND_BINS = np.where(EDGES_M[1:] <= BND_MAX_M + 1e-9)[0]
INT_BINS = np.where(EDGES_M[:-1] >= INT_MIN_M - 1e-9)[0]


class Group:
    """Streaming accumulators for one pixel group (unseen / seen / all)."""

    def __init__(self):
        self.n = np.zeros(NB, np.int64)              # foreground pixels per distance bin
        self.e = np.zeros(NB, np.int64)              # errors per distance bin
        self.ent = np.zeros(NB)                      # summed ensemble entropy per bin
        self.n_k = {k: np.zeros(NB, np.int64) for k in FOREGROUND}
        self.e_k = {k: np.zeros(NB, np.int64) for k in FOREGROUND}
        self.conf = np.zeros((len(RADII_M), C, C), np.int64)
        self.kept = np.zeros(len(RADII_M), np.int64)

    def add(self, mask, pred, dist, bidx, entropy_map, sel):
        """sel: boolean (H,W) selecting this group's FOREGROUND pixels."""
        if not sel.any():
            return
        err = sel & (pred != mask)
        self.n += np.bincount(bidx[sel], minlength=NB)
        self.e += np.bincount(bidx[err], minlength=NB)
        self.ent += np.bincount(bidx[sel], weights=entropy_map[sel].astype(float), minlength=NB)
        for k in FOREGROUND:
            sk = sel & (mask == k)
            self.n_k[k] += np.bincount(bidx[sk], minlength=NB)
            self.e_k[k] += np.bincount(bidx[sk & (pred != mask)], minlength=NB)
        for ri, N in enumerate(RADII_M):
            keep = sel & (dist > N)
            if not keep.any():
                continue
            self.kept[ri] += int(keep.sum())
            self.conf[ri] += conf_over(mask[keep], pred[keep])

    def summary(self) -> dict:
        def rate(e, n):
            n = np.where(n > 0, n, 1)
            return (e / n).tolist()

        def agg(e, n, bins):
            N = int(n[bins].sum())
            return (float(e[bins].sum()) / N if N else float("nan")), N

        within8 = EDGES_M[:-1] < INT_MIN_M
        tot_err = int(self.e.sum())
        per_class = {}
        for k in FOREGROUND:
            be, bn = agg(self.e_k[k], self.n_k[k], BND_BINS)
            ie, in_ = agg(self.e_k[k], self.n_k[k], INT_BINS)
            per_class[STUDENT_CLASSES[k]] = {
                "boundary_error": be, "boundary_n": bn,
                "interior_error": ie, "interior_n": in_,
                "ratio": (be / ie if (ie and np.isfinite(ie) and ie > 0) else None),
            }
        return {
            "n_foreground": int(self.n.sum()),
            "n_errors": tot_err,
            "macro_iou_all_pixels": iou_from_conf(self.conf[0]),
            "trimap_macro_by_radius": {
                f"{(-0.5 if N < 0 else N):.1f}": iou_from_conf(self.conf[ri])["macro_fg"]
                for ri, N in enumerate(RADII_M)
            },
            "trimap_kept_frac": (self.kept / max(1, self.kept[0])).tolist(),
            "error_rate_by_distance": rate(self.e, self.n),
            "n_by_distance": self.n.tolist(),
            "entropy_by_distance": (self.ent / np.where(self.n > 0, self.n, 1)).tolist(),
            "share_of_error_within_8m": (float(self.e[within8].sum()) / tot_err) if tot_err else None,
            "share_of_pixels_within_8m": (float(self.n[within8].sum()) / max(1, int(self.n.sum()))),
            "per_class_boundary_vs_interior": per_class,
        }


class DoseResponse:
    """Error rate and entropy by exposure count, split boundary band vs deep interior."""

    def __init__(self):
        nb = len(EXPOSURE_BINS)
        self.n = np.zeros((nb, 2), np.int64)
        self.e = np.zeros((nb, 2), np.int64)
        self.ent = np.zeros((nb, 2))

    def add(self, cover, fg, mask, pred, bidx, entropy_map):
        band = np.isin(bidx, BND_BINS)
        interior = np.isin(bidx, INT_BINS)
        cov = np.minimum(cover, EXPOSURE_BINS[-1])
        err = pred != mask
        for zi, region in enumerate((band, interior)):
            base = fg & region
            if not base.any():
                continue
            cb = cov[base]
            self.n[:, zi] += np.bincount(cb, minlength=len(EXPOSURE_BINS))
            self.e[:, zi] += np.bincount(cov[base & err], minlength=len(EXPOSURE_BINS))
            self.ent[:, zi] += np.bincount(cb, weights=entropy_map[base].astype(float),
                                           minlength=len(EXPOSURE_BINS))

    def summary(self) -> dict:
        n = np.where(self.n > 0, self.n, 1)
        return {
            "exposure_bins": [f"{b}+" if b == EXPOSURE_BINS[-1] else str(b) for b in EXPOSURE_BINS],
            "regions": ["boundary_band_<=1.5m", "deep_interior_>8m"],
            "n": self.n.tolist(),
            "error_rate": (self.e / n).tolist(),
            "mean_entropy": (self.ent / n).tolist(),
        }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(split, cell, softmax_root, mask_dir, ref_json, root: Path, bounds, train_by_group,
        exposure_dir: Path) -> dict:
    img_ids, dropped = list_val_tiles(root / softmax_root, SEEDS, cell, root / mask_dir)
    print(f"\n[{split}/{cell}] {len(img_ids)} tiles"
          + (f" (dropped {len(dropped)} without a mask)" if dropped else ""))

    unseen, seen, allpx = Group(), Group(), Group()
    dose = DoseResponse()
    wt_unseen, wt_seen = Group(), Group()      # within-tile control
    cover_frac = []
    n_mixed = 0

    for i, iid in enumerate(img_ids, 1):
        stack = load_seed_stack(root / softmax_root, SEEDS, cell, iid)     # (N,C,H,W)
        mask = load_mask(root / mask_dir, iid)
        if mask.shape != stack.shape[-2:]:
            raise ValueError(f"shape mismatch {iid}: mask {mask.shape} vs {stack.shape}")

        cache = exposure_dir / f"{iid}.npy"
        if cache.exists():
            cover = np.load(cache)
        else:
            cover = exposure_map(iid, bounds, train_by_group, mask.shape)
            np.save(cache, cover)

        pred = stack.mean(axis=0).argmax(axis=0).astype(np.int64)
        total_H, _, _ = tile_uncertainty(stack)
        dist = boundary_distance(mask, iid)
        bidx = np.digitize(dist, DIST_BIN_EDGES_M[1:-1])
        fg = mask != 0
        sel_unseen = fg & (cover == 0)
        sel_seen = fg & (cover > 0)

        cover_frac.append(float((cover > 0).mean()))
        allpx.add(mask, pred, dist, bidx, total_H, fg)
        unseen.add(mask, pred, dist, bidx, total_H, sel_unseen)
        seen.add(mask, pred, dist, bidx, total_H, sel_seen)
        dose.add(cover, fg, mask, pred, bidx, total_H)

        # within-tile control: only tiles holding a usable amount of both
        if sel_unseen.sum() >= 1000 and sel_seen.sum() >= 1000:
            n_mixed += 1
            wt_unseen.add(mask, pred, dist, bidx, total_H, sel_unseen)
            wt_seen.add(mask, pred, dist, bidx, total_H, sel_seen)

        if i % 25 == 0 or i == len(img_ids):
            print(f"  {i}/{len(img_ids)} tiles", flush=True)

    out = {
        "split": split, "cell": cell, "n_tiles": len(img_ids), "seeds": SEEDS,
        "mean_tile_coverage_by_training": float(np.mean(cover_frac)),
        "distance_edges_m": EDGES_M.tolist(),
        "all_pixels": allpx.summary(),
        "unseen": unseen.summary(),
        "seen": seen.summary(),
        "dose_response": dose.summary(),
        "within_tile_control": {
            "n_tiles": n_mixed,
            "unseen": wt_unseen.summary(),
            "seen": wt_seen.summary(),
        },
    }

    # --- sanity gate against the stored manuscript numbers ---
    ref_path = root / ref_json
    gate = {"reference": ref_json}
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        ref_curve = ref["recovery_trimap"]["ensemble_iou_per_radius"]
        ref_radii = ref["recovery_trimap"]["radii_m"]
        ev = ref["error_vs_distance"]
        n_ref = np.array(ev["n_foreground"], float)
        e_ref = n_ref * np.array(ev["error_rate_foreground"], float)
        w8 = np.array(ev["edges_m"][:-1]) < INT_MIN_M
        got = out["all_pixels"]
        checks = {
            "macro_iou_no_exclusion": (ref_curve[0]["macro_fg"], got["macro_iou_all_pixels"]["macro_fg"]),
            "macro_iou_4m_exclusion": (ref_curve[ref_radii.index(4.0)]["macro_fg"],
                                       got["trimap_macro_by_radius"]["4.0"]),
            "share_of_error_within_8m": (float(e_ref[w8].sum() / e_ref.sum()),
                                         got["share_of_error_within_8m"]),
        }
        gate["checks"] = {k: {"expected": a, "got": b, "abs_diff": abs(a - b), "pass": abs(a - b) < 5e-4}
                          for k, (a, b) in checks.items()}
        gate["all_pass"] = all(c["pass"] for c in gate["checks"].values())
    else:
        gate["all_pass"] = None
        gate["note"] = "reference json not found; gate skipped"
    out["sanity_gate"] = gate
    return out


def report(res: dict) -> None:
    g = res["sanity_gate"]
    tag = {True: "PASS", False: "FAIL", None: "SKIP"}[g.get("all_pass")]
    print(f"\n{'='*78}\n{res['split']}/{res['cell']}  ({res['n_tiles']} tiles)   sanity gate: {tag}")
    for k, c in g.get("checks", {}).items():
        print(f"   {k:28s} expected {c['expected']:.4f}  got {c['got']:.4f}  "
              f"{'ok' if c['pass'] else 'MISMATCH'}")
    print(f"  mean tile coverage by training tiles: {100*res['mean_tile_coverage_by_training']:.1f}%")

    u, s, a = res["unseen"], res["seen"], res["all_pixels"]
    print(f"\n  {'':34s} {'UNSEEN':>12s} {'SEEN':>12s} {'ALL':>12s}")
    print(f"  {'foreground pixels':34s} {u['n_foreground']:12,d} {s['n_foreground']:12,d} "
          f"{a['n_foreground']:12,d}")
    for lbl, key in [("macro foreground IoU", None), ("overall error rate", None),
                     ("share of error within 8 m", "share_of_error_within_8m")]:
        if key:
            vals = [x[key] for x in (u, s, a)]
            print(f"  {lbl:34s} " + " ".join(f"{100*v:11.1f}%" if v is not None else f"{'n/a':>12s}"
                                             for v in vals))
        elif lbl.startswith("macro"):
            vals = [x["macro_iou_all_pixels"]["macro_fg"] for x in (u, s, a)]
            print(f"  {lbl:34s} " + " ".join(f"{100*v:11.2f}%" for v in vals))
        else:
            vals = [x["n_errors"] / max(1, x["n_foreground"]) for x in (u, s, a)]
            print(f"  {lbl:34s} " + " ".join(f"{100*v:11.2f}%" for v in vals))

    print(f"\n  boundary band (<=1.5 m) vs deep interior (>8 m) error rate, per class:")
    print(f"   {'class':14s} {'UNSEEN bnd':>11s} {'UNSEEN int':>11s} {'ratio':>7s} "
          f"{'SEEN bnd':>10s} {'SEEN int':>10s} {'ratio':>7s}")
    for k in FOREGROUND:
        nm = STUDENT_CLASSES[k]
        cu = u["per_class_boundary_vs_interior"][nm]
        cs = s["per_class_boundary_vs_interior"][nm]
        f = lambda x: "n/a" if x is None or not np.isfinite(x) else f"{100*x:.2f}%"
        r = lambda x: "n/a" if x is None or not np.isfinite(x) else f"{x:.0f}x"
        print(f"   {nm:14s} {f(cu['boundary_error']):>11s} {f(cu['interior_error']):>11s} "
              f"{r(cu['ratio']):>7s} {f(cs['boundary_error']):>10s} {f(cs['interior_error']):>10s} "
              f"{r(cs['ratio']):>7s}")

    d = res["dose_response"]
    print(f"\n  dose-response (error rate by number of covering training tiles):")
    print(f"   {'exposure':>9s} {'bnd err':>9s} {'bnd n':>12s} {'int err':>9s} {'int n':>12s}")
    for i, b in enumerate(d["exposure_bins"]):
        print(f"   {b:>9s} {100*d['error_rate'][i][0]:8.2f}% {d['n'][i][0]:12,d} "
              f"{100*d['error_rate'][i][1]:8.2f}% {d['n'][i][1]:12,d}")

    wt = res["within_tile_control"]
    if wt["n_tiles"]:
        wu, ws = wt["unseen"], wt["seen"]
        print(f"\n  within-tile control ({wt['n_tiles']} tiles holding both groups):")
        print(f"   overall error rate   unseen {100*wu['n_errors']/max(1,wu['n_foreground']):.2f}%   "
              f"seen {100*ws['n_errors']/max(1,ws['n_foreground']):.2f}%")
        print(f"   error within 8 m     unseen {100*wu['share_of_error_within_8m']:.1f}%   "
              f"seen {100*ws['share_of_error_within_8m']:.1f}%")

    print(f"\n  trimap macro IoU by excluded band (m), unseen pixels only:")
    tm = u["trimap_macro_by_radius"]
    print("   " + "  ".join(f"{k}m={100*v:.2f}" for k, v in list(tm.items())[:7]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="restrict to one run, e.g. val:stage3_clsbal")
    ap.add_argument("--out-dir", default="analysis/leakage_check")
    args = ap.parse_args()

    root = find_repo_root()
    out_dir = root / args.out_dir
    exposure_dir = out_dir / "exposure"
    exposure_dir.mkdir(parents=True, exist_ok=True)

    print("reading tile footprints from GeoTIFF transforms ...", flush=True)
    bounds = read_bounds(root, cache=out_dir / "tile_bounds.json")
    train_by_group = {}
    for tid, (split, *_) in bounds.items():
        if split == "train":
            train_by_group.setdefault(crs_group(tid), []).append(tid)
    print(f"  {len(bounds)} tiles; training tiles per CRS group: "
          f"{ {g: len(v) for g, v in train_by_group.items()} }", flush=True)

    runs = RUNS
    if args.only:
        want = tuple(args.only.split(":"))
        runs = [r for r in RUNS if (r[0], r[1]) == want]
        if not runs:
            raise SystemExit(f"no run matches {args.only}")

    results = {}
    for split, cell, sroot, mdir, ref in runs:
        if not (root / sroot).exists():
            print(f"[skip] {split}/{cell}: {sroot} not found")
            continue
        res = run(split, cell, sroot, mdir, ref, root, bounds, train_by_group, exposure_dir)
        results[f"{split}:{cell}"] = res
        report(res)
        (out_dir / f"leakage_check_{split}_{cell}.json").write_text(json.dumps(res, indent=2))

    (out_dir / "leakage_check_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_dir}/leakage_check_*.json")


if __name__ == "__main__":
    main()
