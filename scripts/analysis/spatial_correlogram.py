#!/usr/bin/env python3
"""
Multivariate Mantel correlogram: how far apart must two tiles be before they stop looking alike?

WHY. Block size and buffer width in `scripts/data_prep/build_spatial_split.py` were chosen by eye.
Roberts et al. (2017) require blocks LARGER than the range over which nearby ground stays similar,
and a buffer AT LEAST that range; and if the range exceeds half a site's extent, no independent
split of that site exists however it is cut. Nobody measured the range, so none of those three
choices is currently justified. This measures it.

METHOD. The standard multivariate Mantel correlogram (Legendre & Legendre 2012, sec. 13.1.6; the
method implemented by `vegan::mantel.correlog`), computed per site because the sites are ~55 km
apart in two different coordinate systems:

  1. Build a response distance matrix Y over tiles from a multivariate per-tile descriptor.
  2. Split geographic distance into classes of fixed width.
  3. For each class k, build a binary model matrix D_k (1 where a pair falls in that class).
  4. Mantel statistic = Pearson r between the vectorised Y and D_k, SIGN REVERSED so that positive
     values mean positive spatial autocorrelation (vegan's convention).
  5. Test by permuting tile order; correct across classes with progressive Holm.
  6. The range is the first class at which autocorrelation stops being significantly positive.

Only classes up to HALF the maximum geographic distance are computed, per Legendre & Legendre:
beyond that, pair counts fall away and the statistic is unreliable.

Two descriptors, mirroring Kattenborn et al. (2022), who run one correlogram on the response and
one on imagery latents (`code/2_correlogram_02_response_v1.R`, `..._01_predictors_3.R`, both via
`ncf::correlog`):

  --descriptor composition  per-tile class-composition vector, Hellinger-transformed
                            (Legendre & Gallagher 2001, the standard transform for compositional
                            data before Euclidean distance). Needs only the masks: no model, no GPU.
  --descriptor spectral     per-tile band statistics (mean/sd/percentiles per band, plus NDVI when
                            a 4th band exists), standardised. Needs only the imagery.

Agreement between the two is the robustness check. Neither requires training anything.

Run:
    PYTHONPATH=. python scripts/analysis/spatial_correlogram.py --descriptor composition
    PYTHONPATH=. python scripts/analysis/spatial_correlogram.py --descriptor spectral
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

# Metres per degree: ONE definition, in geoseg/geo.py. These two literals appeared in seven
# files with no derivation, and terrain_separability used the LONGITUDE constant for the
# latitude direction.
from geoseg.geo import M_PER_DEG_LAT, M_PER_DEG_LON_EQ  # noqa: E402,F401

FOREGROUND = {1: "Forest", 2: "Grassland", 3: "Cropland", 4: "Settlement", 5: "Seminatural"}


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "data").is_dir() and (parent / "artifacts").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def site_of(tile_id: str) -> str:
    return tile_id.split("_")[0]


# ---------------------------------------------------------------------------
# Geometry: tile centroids in METRES, per site
# ---------------------------------------------------------------------------
def centroids_m(root: Path, split_root: Path, want_bounds: bool = False) -> dict:
    """{tile_id: (site, x_m, y_m)} with a local metric frame per site.

    With want_bounds, returns (site, cx, cy, left, bottom, right, top) so the caller can measure a
    site's true FOOTPRINT extent. Centroid spread understates it by one tile width, which matters:
    it is what decides whether a site can be partitioned at all.

    Degrees are converted at the site's mean latitude, so a degree of longitude and a degree of
    latitude do not get treated as the same distance -- the upland sites are anisotropic
    (0.64 x 0.51 m pixels), and ignoring that would distort every distance in the correlogram.
    """
    raw = {}
    for split in ("train", "val", "test"):
        d = split_root / split / "images"
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.tif")):
            with rasterio.open(f) as src:
                b = src.bounds
                raw[f.stem] = (src.crs.is_geographic if src.crs else False,
                               (b.left + b.right) / 2, (b.bottom + b.top) / 2,
                               b.left, b.bottom, b.right, b.top)
    out = {}
    for site in sorted({site_of(k) for k in raw}):
        ids = [k for k in raw if site_of(k) == site]
        geo = raw[ids[0]][0]
        lat0 = np.mean([raw[k][2] for k in ids]) if geo else None
        mx = M_PER_DEG_LON_EQ * math.cos(math.radians(lat0)) if geo else 1.0
        my = M_PER_DEG_LAT if geo else 1.0
        for k in ids:
            _, x, y, l, b_, r, t = raw[k]
            out[k] = ((site, x * mx, y * my) if not want_bounds
                      else (site, x * mx, y * my, l * mx, b_ * my, r * mx, t * my))
    return out


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------
def composition_descriptor(split_root: Path, ids: list) -> np.ndarray:
    """Hellinger-transformed class proportions (Legendre & Gallagher 2001)."""
    rows = []
    for tid in ids:
        for split in ("train", "val", "test"):
            p = split_root / split / "masks" / f"{tid}.png"
            if p.exists():
                m = np.array(Image.open(p).convert("L"))
                break
        counts = np.array([(m == c).sum() for c in FOREGROUND], float)
        tot = counts.sum()
        rows.append(counts / tot if tot > 0 else counts)
    P = np.array(rows)
    return np.sqrt(P)                       # Hellinger: sqrt of relative abundances


def spectral_descriptor(split_root: Path, ids: list) -> np.ndarray:
    """Per-band mean/sd/10th/90th percentile, plus NDVI mean/sd where a NIR band exists."""
    rows = []
    for tid in ids:
        for split in ("train", "val", "test"):
            p = split_root / split / "images" / f"{tid}.tif"
            if p.exists():
                break
        with rasterio.open(p) as src:
            a = src.read(out_shape=(src.count, 128, 128)).astype(np.float32)
        a = np.where(np.isfinite(a), a, np.nan)
        feats = []
        for band in a:
            v = band[np.isfinite(band)]
            if v.size == 0:
                feats += [0.0, 0.0, 0.0, 0.0]
            else:
                feats += [float(v.mean()), float(v.std()),
                          float(np.percentile(v, 10)), float(np.percentile(v, 90))]
        if a.shape[0] >= 4:
            red, nir = a[0], a[3]
            with np.errstate(invalid="ignore", divide="ignore"):
                ndvi = (nir - red) / (nir + red)
            v = ndvi[np.isfinite(ndvi)]
            feats += [float(v.mean()), float(v.std())] if v.size else [0.0, 0.0]
        rows.append(feats)
    X = np.array(rows, float)
    X = np.nan_to_num(X)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - X.mean(axis=0)) / sd        # standardise so no band dominates


# ---------------------------------------------------------------------------
# Mantel correlogram
# ---------------------------------------------------------------------------
def mantel_correlogram(Z: np.ndarray, xy: np.ndarray, increment: float,
                       n_perm: int, rng: np.random.Generator) -> list:
    n = len(Z)
    iu = np.triu_indices(n, k=1)
    Yfull = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)     # response distances (n x n)
    Y = Yfull[iu]
    G = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)       # geographic distances
    g = G[iu]

    # Legendre & Legendre: only classes up to half the maximum distance are interpretable.
    cutoff = g.max() / 2.0
    edges = np.arange(0.0, cutoff + increment, increment)

    Yc = Y - Y.mean()
    Yss = np.sqrt((Yc ** 2).sum())
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (g > lo) & (g <= hi)
        npair = int(sel.sum())
        if npair < 30:                       # too few pairs to be meaningful
            continue
        d = sel.astype(float)
        dc = d - d.mean()
        r = float((Yc * dc).sum() / (Yss * np.sqrt((dc ** 2).sum())))
        r = -r                               # vegan convention: positive = positive autocorrelation

        # Permutation test. Standard Mantel: permute the LABELS of the precomputed distance matrix,
        # i.e. reorder its rows and columns together. Mathematically identical to recomputing the
        # distances from permuted data, but O(n^2) per permutation instead of O(n^2 d), which is what
        # makes n_perm = 9999 affordable and removes the right-censoring that n_perm = 199 imposed
        # (with 199 the smallest achievable p is 1/200, and progressive Holm then kills any class
        # sitting at that floor by k = 10 regardless of the true signal).
        ge = 0
        for _ in range(n_perm):
            perm = rng.permutation(n)
            Yp = Yfull[np.ix_(perm, perm)][iu]
            Ypc = Yp - Yp.mean()
            rp = -float((Ypc * dc).sum() / (np.sqrt((Ypc ** 2).sum()) * np.sqrt((dc ** 2).sum())))
            if abs(rp) >= abs(r):
                ge += 1
        out.append({"lo_m": float(lo), "hi_m": float(hi), "centre_m": float((lo + hi) / 2),
                    "n_pairs": npair, "mantel_r": r, "p_raw": (ge + 1) / (n_perm + 1)})

    # progressive Holm correction across classes, as vegan::mantel.correlog does
    for i, rec in enumerate(out):
        rec["p_holm"] = float(min(1.0, max(r_["p_raw"] * (i + 1 - j)
                                           for j, r_ in enumerate(out[:i + 1]))))
    return out


def read_range(recs: list) -> float | None:
    """First distance class at which positive autocorrelation stops being significant."""
    for rec in recs:
        if not (rec["mantel_r"] > 0 and rec["p_holm"] < 0.05):
            return rec["centre_m"]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptor", choices=["composition", "spectral"], default="composition")
    ap.add_argument("--split-root", default="data/biodiversity_split",
                    help="tile pool (the legacy three-directory layout is used only as a container)")
    ap.add_argument("--increment", type=float, default=100.0, help="distance class width, metres")
    ap.add_argument("--n-perm", type=int, default=199)
    ap.add_argument("--max-tiles", type=int, default=600,
                    help="cap per site; Mantel is O(n^2) per permutation, and a random subsample is "
                         "an unbiased estimate of the same correlogram")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="analysis/correlogram")
    args = ap.parse_args()

    root = find_repo_root()
    rng = np.random.default_rng(args.seed)
    print(f"reading tile centroids ({args.descriptor} descriptor) ...", flush=True)
    cent = centroids_m(root, root / args.split_root, want_bounds=True)

    results = {}
    for site in sorted({v[0] for v in cent.values()}):
        ids = sorted([k for k, v in cent.items() if v[0] == site])
        if len(ids) > args.max_tiles:
            ids = sorted(rng.choice(ids, args.max_tiles, replace=False).tolist())
        xy = np.array([[cent[k][1], cent[k][2]] for k in ids], float)
        xy -= xy.mean(axis=0)
        # FOOTPRINT extent over ALL of the site's tiles (not just the subsample, not centroids):
        # this is what Roberts' half-extent criterion is defined against.
        site_all = [k for k, v in cent.items() if v[0] == site]
        extent = (max(cent[k][5] for k in site_all) - min(cent[k][3] for k in site_all),
                  max(cent[k][6] for k in site_all) - min(cent[k][4] for k in site_all))
        print(f"\n### {site}: {len(ids)} tiles, extent {extent[0]:.0f} x {extent[1]:.0f} m", flush=True)

        Z = (composition_descriptor(root / args.split_root, ids) if args.descriptor == "composition"
             else spectral_descriptor(root / args.split_root, ids))
        recs = mantel_correlogram(Z, xy, args.increment, args.n_perm, rng)
        rng_m = read_range(recs)
        peak_r = max((rec["mantel_r"] for rec in recs), default=float("nan"))
        half = max(extent) / 2

        print(f"  {'class (m)':>14} {'pairs':>8} {'Mantel r':>10} {'p(Holm)':>9}")
        for rec in recs[:14]:
            star = " *" if (rec["mantel_r"] > 0 and rec["p_holm"] < 0.05) else ""
            print(f"  {rec['lo_m']:6.0f}-{rec['hi_m']:6.0f} {rec['n_pairs']:8d} "
                  f"{rec['mantel_r']:10.4f} {rec['p_holm']:9.3f}{star}")
        print(f"  -> autocorrelation range: "
              f"{'>' + str(int(recs[-1]['hi_m'])) + ' m (never dies out)' if rng_m is None else f'{rng_m:.0f} m'}")
        print(f"  -> half-extent of this site: {half:.0f} m")
        ratio = (max(extent) / rng_m) if rng_m else float("inf")
        if rng_m is None or rng_m > half:
            print(f"  -> Roberts half-extent rule FAILS: range exceeds half the extent, so no "
                  f"independent within-site split of {site} exists however it is cut.")
        elif ratio < 3.0:
            print(f"  -> Roberts half-extent rule passes ({rng_m:.0f} < {half:.0f} m), BUT the site "
                  f"is only {ratio:.1f}x the range along its long axis. Fitting "
                  f"train|buffer|test needs >=3x ({3*rng_m:.0f} m), so no buffered partition fits: "
                  f"hold {site} out whole.")
        else:
            print(f"  -> splittable ({ratio:.1f}x the range). Blocks must exceed {rng_m:.0f} m; "
                  f"buffer at least {rng_m:.0f} m.")
        print(f"  -> PEAK Mantel r = {peak_r:.4f}"
              + ("   (>= 0.10: practically meaningful)" if peak_r >= 0.10 else
                 "   (< 0.10: statistically detectable but practically negligible -- "
                 "per the pre-registered rule the buffer stays anchored on tile geometry)"))
        results[site] = {"n_tiles_used": len(ids), "n_tiles_total": len(site_all),
                         "subsampled": len(ids) < len(site_all),
                         "extent_m": list(extent), "extent_basis": "full GeoTIFF footprints",
                         "half_extent_m": half, "range_m": rng_m, "peak_mantel_r": peak_r,
                         "classes": recs}

    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"correlogram_{args.descriptor}.json"
    p.write_text(json.dumps({"descriptor": args.descriptor, "increment_m": args.increment,
                             "n_perm": args.n_perm, "sites": results}, indent=2))
    # relative_to raises for a path outside the repo, AFTER the result is written:
    # exit 1 on a success, from a cosmetic print.
    try:
        _shown = p.relative_to(root)
    except ValueError:
        _shown = p
    print(f"\nwrote {_shown}")


if __name__ == "__main__":
    main()
