"""The single source of truth for this project's ground geometry constants.

WHY THIS MODULE EXISTS. Until 2026-07-26 the per-site ground sample distance was hand-typed as a
literal dict in several files (seed_disagreement, boundary_exposure,
figures/boundary_limited_error) and the metres-per-degree constants appeared in SEVEN
(analysis/utils, data_prep/build_spatial_split, analysis/block_phase_sweep,
analysis/report_class_support, analysis/accuracy_vs_separation, analysis/spatial_correlogram,
data_prep/assert_no_split_leakage). Every one of them is derivable from the GeoTIFF transforms, so
none of them needed to be typed at all — and one of the seven, terrain_separability, used the
LONGITUDE constant for the latitude direction.

These values are load-bearing: `boundary_distance` returns metres, the registered band is a strict
`< 8.0 m`, and the upland sites are anisotropic, so a wrong figure silently rescales every band at
191 of 1,730 tiles.

    ON THE VALUE OF THE CONSTANTS, AND WHY THEY ARE NOT THE MOST ACCURATE ONES AVAILABLE

    M_PER_DEG_LON_EQ and M_PER_DEG_LAT below are the SPHERICAL approximation. The ellipsoidal
    (WGS84) figure is larger by 1/sqrt(1 - e^2 sin^2 phi) = 1.00208 at 51.5 N, so the upland x
    ground sample distances derived here are 0.20-0.22% low:

        site      registered (spherical)   geodesic (pyproj.Geod, WGS84)
        ireland1  0.641 m                  0.6423 m
        ireland2  0.634 m                  0.6354 m

    The geodesic values are the more correct ones. They are deliberately NOT adopted, because
    GSD_BY_SITE feeds `artifacts/boundary_band_denominators.json`, which was registered BEFORE any
    model was trained on this split. Adopting them would move the Test B band area share from
    26.4796% to 26.430% — 0.05 pp, far below the width of any interval it appears in — at the cost
    of the one property that makes a registered denominator worth having, namely that it was fixed
    in advance. A pre-registered quantity is not re-derived on the eve of the campaign for a
    fourth-significant-figure gain.

    Revisit after the campaign, deliberately, as a documented change with the denominators
    regenerated — not as a side effect of tidying constants.

Verify the relationship rather than trusting this comment:
    PYTHONPATH=. python geoseg/geo.py
"""
from __future__ import annotations

import math

# Spherical metres per degree. See the module docstring for why these and not the ellipsoidal pair.
M_PER_DEG_LON_EQ = 111320.0   # longitude, at the equator; scale by cos(latitude)
M_PER_DEG_LAT = 111132.0      # latitude; very nearly constant, NOT interchangeable with the above

# Per-site ground sample distance, (y, x) metres per pixel, measured from the GeoTIFF transforms.
# The inland site is projected (UTM 29N) and isotropic; both upland sites are geographic
# (EPSG:4326) and ANISOTROPIC, so a single 0.5 m figure is wrong for 191 of 2,143 tiles.
# Pass sampling=(gsd_y, gsd_x) to scipy.ndimage.distance_transform_edt so distances come out in
# metres directly.
GSD_BY_SITE: dict[str, tuple[float, float]] = {
    "biodiversity": (0.500, 0.500),
    "ireland1":     (0.515, 0.641),
    "ireland2":     (0.515, 0.634),
}


def site_of(tile_id: str) -> str:
    """The site prefix of a tile id, e.g. 'ireland2_0060' -> 'ireland2'."""
    return str(tile_id).split("_")[0]


def gsd_for(tile_id: str) -> tuple[float, float]:
    """(gsd_y, gsd_x) in metres for a tile id, keyed on its site prefix.

    RAISES on an unknown site. The previous implementation returned (0.5, 0.5) for anything it did
    not recognise, which is the inland site's isotropic value: a mis-parsed or newly added upland
    tile would have had every boundary distance silently rescaled by up to 28% in x, with no error
    and no warning. A distance in metres is the registered unit of the primary statistic; it does
    not get a default.
    """
    site = site_of(tile_id)
    try:
        return GSD_BY_SITE[site]
    except KeyError:
        raise KeyError(
            f"no ground sample distance registered for site '{site}' (tile '{tile_id}'). "
            f"Known sites: {sorted(GSD_BY_SITE)}. Add it to geoseg/geo.py, measured from the "
            f"GeoTIFF transform — do not let it fall back to the inland 0.5 m."
        ) from None


def m_per_deg_lon(lat_deg: float) -> float:
    """Metres per degree of longitude at a given latitude."""
    return M_PER_DEG_LON_EQ * math.cos(math.radians(lat_deg))


def deg_per_m_lon(lat_deg: float) -> float:
    """Degrees of longitude per metre at a given latitude."""
    return 1.0 / m_per_deg_lon(lat_deg)


def deg_per_m_lat() -> float:
    """Degrees of latitude per metre."""
    return 1.0 / M_PER_DEG_LAT


def _self_test() -> int:
    """Check the registered constants against the GeoTIFF transforms they claim to come from."""
    from pathlib import Path
    import glob
    ok = True
    try:
        import rasterio
        from pyproj import Geod
    except ImportError:
        print("rasterio/pyproj unavailable; skipping the raster check")
        return 0

    root = Path(__file__).resolve().parents[1]
    geod = Geod(ellps="WGS84")
    print("registered GSD vs the GeoTIFF transforms (geodesic):")
    for site, (ry, rx) in GSD_BY_SITE.items():
        files = sorted(glob.glob(str(root / f"data/biodiversity_raw/images/{site}_*.tif")))
        if not files:
            print(f"  {site:<14} no rasters on disk, skipped")
            continue
        with rasterio.open(files[0]) as ds:
            t = ds.transform
            cx, cy = t.c + t.a * ds.width / 2.0, t.f - (-t.e) * ds.height / 2.0
            if ds.crs and ds.crs.is_geographic:
                _, _, gx = geod.inv(cx, cy, cx + t.a, cy)
                _, _, gy = geod.inv(cx, cy, cx, cy + (-t.e))
                sx = t.a * m_per_deg_lon(cy)          # what THIS module's constants give
                sy = (-t.e) * M_PER_DEG_LAT
            else:
                gx = sx = t.a
                gy = sy = -t.e
        dx = 100.0 * (gx - rx) / rx
        # The registered value must match this module's own spherical derivation, not the geodesic.
        # TOLERANCE, and why it is what it is. GSD_BY_SITE is registered to 3 decimal places, so the
        # most a correct value can differ from the derivation is half a unit in the last place:
        # 5e-4. Testing `< 5e-4` therefore passed the upland latitude by 1.6e-14 -- the registered
        # 0.515 against a derived 0.51450 is EXACTLY the tolerance, and the check survived on
        # floating-point luck rather than on being right. `<=` plus a small epsilon states the real
        # property: the registered value is the derivation rounded to 3 dp.
        tol = 5e-4 + 1e-9
        good = abs(sx - rx) <= tol and abs(sy - ry) <= tol
        good &= round(sx, 3) == rx and round(sy, 3) == ry
        ok &= good
        print(f"  {site:<14} registered (y={ry:.4f}, x={rx:.4f})  "
              f"spherical (y={sy:.4f}, x={sx:.4f})  geodesic (y={gy:.4f}, x={gx:.4f})  "
              f"x differs from geodesic by {dx:+.2f}%  [{'ok' if good else 'DRIFT'}]")

    try:
        gsd_for("nosuchsite_0001")
        print("  gsd_for did NOT raise on an unknown site  [FAIL]")
        ok = False
    except KeyError:
        print("  gsd_for raises on an unknown site rather than defaulting to 0.5 m  [ok]")

    if M_PER_DEG_LAT == M_PER_DEG_LON_EQ:
        print("  latitude and longitude constants are equal  [FAIL]")
        ok = False

    print("\nSELF-TEST PASSED" if ok else "\nSELF-TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
