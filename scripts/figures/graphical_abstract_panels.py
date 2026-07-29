#!/usr/bin/env python3
"""The three raster panels of the graphical abstract.

REBUILT 2026-07-29. Every element of the previous version was withdrawn: its chip
(biodiversity_0957) is a TRAINING chip under the current split, so the abstract showed the model
marking its own homework; its badges carried the leakage-inflated 90.8% mean IoU and the retracted
"<1% inside, 42% at boundaries" error share; and its caption stated the label-quality-ceiling claim
that `docs/DO_NOT_ADD.md` forbids. It also rebuilt from `sonic/results/`, which is not staged here.

The three panels are ONE Test A chip seen three ways, so the abstract is an argument rather than a
pipeline:

  1. the imagery -- the two grassland classes are a distinction of management, not appearance
  2. where the model is wrong -- whole parcels flipped between them, not rims
  3. how often the two classes actually meet -- grassland's 8 m band to forest against to
     semi-natural, both drawn on the same picture so the contrast is one glance

Chip biodiversity_1078, seed 47, stage1_baseline. It is in the 90-chip Test A scoring subset, is
fully labelled, holds over a hectare of each of the two classes, and is the chip class_seam.py
uses -- so the abstract and that figure show the same ground. The split is asserted below rather
than trusted.

Numbers that go on the badges are in the ledger: 46.7% (pair share of foreground error), 0.6% and
21.4% (grassland within 8 m of semi-natural and of forest).

Writes figures/graphical_abstract/ga_panel{1,2,3}*.png, read by graphical_abstract_tikz.tex.

Run:
  python scripts/figures/graphical_abstract_panels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "geoseg").is_dir() and (parent / "config").is_dir():
            return parent
    raise RuntimeError("repo root not found")


REPO = find_repo_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from geoseg.taxonomy import STUDENT_PALETTE  # noqa: E402

SPLIT_TAG = "f1"
TILE = "biodiversity_1078"
SEED = 47
CELL = "stage1_baseline"
NEAR_M = 8.0
PX_M = 0.5
FOREST, GRASS, SEMI = 1, 2, 5
SIDE = 1200                       # output side in px; panels are square

PALETTE = np.array(STUDENT_PALETTE, dtype=np.uint8)
# Panel 2, the two error directions. Same two colours as two_grasslands_qualitative, so a reader
# who meets the abstract first is not relearning them at that figure.
ERR_G2S = (178, 24, 43)
ERR_S2G = (84, 39, 136)
# Panel 3, the two bands. One cool, one warm, and neither is a class colour.
BAND_FOREST = (59, 42, 107)
BAND_SEMI = (232, 106, 20)
WASH = 0.52

OUT = REPO / "figures/graphical_abstract"


def assert_test_chip() -> None:
    """Refuse to build from a chip the model was trained on.

    This is the defect that shipped last time. The manifest is the authority; the scoring subset is
    checked too, so the abstract cannot show ground the reported numbers exclude.
    """
    manifest = json.loads((REPO / f"artifacts/spatial_split_manifest_{SPLIT_TAG}.json").read_text())
    assignment = manifest.get("assignment", manifest)
    split = assignment.get(TILE)
    if split != "test":
        raise SystemExit(
            f"{TILE} is '{split}' in the split manifest, not 'test'. The graphical abstract would "
            f"show the model predicting on ground it was trained on, which is exactly the defect "
            f"the previous version shipped with.")
    subset = json.loads((REPO / f"artifacts/scoring_subset_{SPLIT_TAG}.json").read_text())
    if TILE not in set(subset["splits"]["test"]["tiles"]):
        raise SystemExit(f"{TILE} is not in the 90-chip scoring subset the paper's numbers use.")


def read_mask() -> np.ndarray:
    return np.array(Image.open(REPO / f"data/split_{SPLIT_TAG}/test/masks/{TILE}.png").convert("L"))


def read_rgb() -> np.ndarray:
    with rasterio.open(REPO / f"data/split_{SPLIT_TAG}/test/images/{TILE}.tif") as src:
        a = src.read([1, 2, 3]).astype(np.float32)
    out = np.empty_like(a)
    for b in range(3):
        lo, hi = np.percentile(a[b], [2, 98])
        out[b] = np.clip((a[b] - lo) / (hi - lo), 0, 1)
    return (out.transpose(1, 2, 0) * 255).round().astype(np.uint8)


def read_prediction() -> np.ndarray:
    p = REPO / f"analysis/panel_root/seed{SEED}/analysis/seed_softmax/{CELL}/seed{SEED}/{TILE}.npy"
    if not p.is_file():
        raise SystemExit(f"prediction not staged: {p}")
    return np.load(p).argmax(axis=0).astype(np.uint8)


def washed(mask: np.ndarray) -> np.ndarray:
    return (PALETTE[mask].astype(np.float32) * (1 - WASH) + 255 * WASH).astype(np.uint8)


def save(arr: np.ndarray, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).resize((SIDE, SIDE), Image.NEAREST).save(OUT / name)
    print(f"  wrote {name}")


def main() -> int:
    assert_test_chip()
    mask = read_mask()
    pred = read_prediction()
    pred[mask == 0] = 0

    # 1 -- the imagery. Licensed; the credit is placed by the TikZ assembly, per the Airbus EULA.
    save(read_rgb(), "ga_panel1_rgb.png")

    # 2 -- the pair's error over a washed reference, so the flipped parcels read as blocks.
    p2 = washed(mask)
    p2[(mask == GRASS) & (pred == SEMI)] = ERR_G2S
    p2[(mask == SEMI) & (pred == GRASS)] = ERR_S2G
    save(p2, "ga_panel2_error.png")

    # 3 -- how much of grassland's ground lies within 8 m of each of the two classes, both bands on
    # one picture. Semi-natural is drawn last so the sparser band cannot be hidden by the denser.
    d_forest = ndimage.distance_transform_edt(mask != FOREST, sampling=(PX_M, PX_M))
    d_semi = ndimage.distance_transform_edt(mask != SEMI, sampling=(PX_M, PX_M))
    g = mask == GRASS
    band_f, band_s = g & (d_forest < NEAR_M), g & (d_semi < NEAR_M)
    p3 = washed(mask)
    p3[band_f] = BAND_FOREST
    p3[band_s] = BAND_SEMI
    save(p3, "ga_panel3_seam.png")

    pair = ((mask == GRASS) | (mask == SEMI)).sum()
    pair_err = int(((mask == GRASS) & (pred == SEMI)).sum()
                   + ((mask == SEMI) & (pred == GRASS)).sum())
    print(f"{TILE}, seed {SEED}, {CELL}: verified Test A and in the scoring subset")
    print(f"  pair error on this chip: {pair_err:,} px ({100 * pair_err / pair:.1f}% of its ground)")
    print(f"  grassland within 8 m of forest {100 * band_f.sum() / g.sum():.1f}%, "
          f"of semi-natural {100 * band_s.sum() / g.sum():.2f}% (pooled: 21.4%, 0.6%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
