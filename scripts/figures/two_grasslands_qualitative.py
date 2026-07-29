#!/usr/bin/env python3
"""
The grassland / semi-natural confusion on the ground, across the range of chips that carry it.

WHY A LADDER AND NOT ONE CHOSEN CHIP. A first version of this figure showed the median chip by
pair-error rate and the chip carrying the largest error volume. An independent checker, given only
the per-chip statistics and asked to argue the selection was cherry-picked, broke it: a chip's
pair-error rate varies by up to three orders of magnitude across the ten seeds, so "the median chip"
is not a stable object, and five innocuous perturbations of the rule -- a minimum class area, seed
median instead of seed mean, 95% instead of 100% labelled, per-seed instead of pooled ranking, rate
instead of volume -- each moved the answer to a chip on which the figure would look worse. Showing
the RANGE removes the choice: the four columns are the smallest, largest and two interior quantiles
of the eligible set, so there is no favourable chip to have picked.

SELECTION, in full, so the caption can state it:
  Population   the 90 Test A scoring chips (artifacts/scoring_subset_f1.json).
  Eligible     fully labelled -- every one of the 512x512 pixels carries a reference class -- and
               holding at least one hectare of BOTH Grassland and Semi-natural reference ground.
               One hectare because that is the size at which the paper measures connected error
               patches; below it a class cannot exhibit the structure the figure is about. Twelve
               of the ninety qualify.
  Columns      those twelve ranked by pair-error rate averaged over the ten seeds, sampled at the
               0th, 33rd, 67th and 100th percentiles -- minimum, maximum and two interior points.
  Seed         each column is shown at the seed whose pair-error rate on THAT chip is closest to
               that chip's own ten-seed mean, so no column is an unrepresentative run of its own
               chip. The seed differs between columns; it is NOT printed on the panel any more
               (the titles carry the ten-seed mean rate alone), so the caption must list them.
               The run prints them, and so does the summary at the foot of main().
  Cell         stage1_baseline, the no-intervention cell of the 2x2 factorial.

Pair-error rate is (Grassland-called-Semi-natural + Semi-natural-called-Grassland) error pixels
divided by Grassland + Semi-natural reference pixels, on reference-foreground pixels only, which is
the convention pooled_confusion.py uses.

Writes:
  figures/two_grasslands_qualitative.pdf

Run:
  python scripts/figures/two_grasslands_qualitative.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Patch
from PIL import Image


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "geoseg").is_dir() and (parent / "config").is_dir():
            return parent
    raise RuntimeError("repo root not found")


REPO = find_repo_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from geoseg.taxonomy import STUDENT_CLASSES, STUDENT_PALETTE  # noqa: E402

SPLIT_TAG = "f1"
CELL = "stage1_baseline"
SEEDS = list(range(42, 52))
G, S = 2, 5
MIN_HA_PX = 40_000          # one hectare at 0.5 x 0.5 m
QUANTILES = (0, 33, 67, 100)
PX_M = 0.5                  # inland site only; asserted below
# PAGE WIDTH. \textwidth in Definitions/mdpi.cls is 394.36 pt = 13.90 cm; \extralength is 4.61 cm,
# so a figure inside \begin{adjustwidth}{-\extralength}{0cm} -- the pattern main.tex already uses --
# gets 18.51 cm = 7.28 in. Built at that width, 1 pt here is 1 pt on the page; built any wider it is
# scaled down by \includegraphics[width=\linewidth] and every label shrinks with it.
FIG_W = 7.28

PALETTE = np.array(STUDENT_PALETTE, dtype=np.uint8)
DISPLAY_NAMES = {"Seminatural": "Semi-natural"}
# The two error directions. Deliberately NOT the class colours: these mark a disagreement between
# the reference and the prediction, not a class, and reusing the class colour for an error would
# make the error panel look like a third label map.
ERR_G2S = "#B2182B"         # reference Grassland, predicted Semi-natural
# Purple, not the blue it was until 2026-07-29: that blue sat beside Settlement's #3B8DF7 and the
# two co-occur in columns (c) and (d), so the class legend and the error legend read as one scale.
# Purple is distinct from all five class colours -- forest pink-red, grassland lime, cropland
# orange, settlement blue, semi-natural yellow -- and from the red of the other error direction.
ERR_S2G = "#542788"         # reference Semi-natural, predicted Grassland
CORRECT_WASH = 0.82         # how far the reference map is washed out behind the error overlay


def set_plot_style() -> None:
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{lmodern}",
        "mathtext.fontset": "stix",
        # One size for the panel letters, the row labels and the legend: they are the three
        # pieces of running text a reader actually uses, and mixing sizes made the figure read as
        # having a hierarchy it does not have.
        "font.size": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


def load_tile_stats() -> dict[int, dict[str, dict]]:
    out = {}
    for s in SEEDS:
        p = REPO / f"analysis/tile_stats/tile_stats_seed{s}_{CELL.replace('stage1_', '')}.json"
        if not p.exists():
            raise SystemExit(
                f"{p} missing. Per-chip statistics are produced on the cluster where the softmax "
                f"dumps live; see the header of this script.")
        out[s] = {r["tile"]: r for r in json.loads(p.read_text())["tiles"]}
    return out


def pair_rate(rec: dict) -> float:
    return (rec["g2s_px"] + rec["s2g_px"]) / (rec["g_px"] + rec["s_px"])


def choose_columns(stats) -> list[dict]:
    ref = stats[SEEDS[0]]
    eligible = [t for t, r in ref.items()
                if r["fg_px"] == 512 * 512 and r["g_px"] >= MIN_HA_PX and r["s_px"] >= MIN_HA_PX]
    if not eligible:
        raise SystemExit("no eligible chips; the selection rule cannot be applied")
    mean_rate = {t: float(np.mean([pair_rate(stats[s][t]) for s in SEEDS])) for t in eligible}
    order = sorted(eligible, key=lambda t: mean_rate[t])

    cols = []
    for q in QUANTILES:
        tile = order[int(round(q / 100 * (len(order) - 1)))]
        per_seed = {s: pair_rate(stats[s][tile]) for s in SEEDS}
        seed = min(SEEDS, key=lambda s: abs(per_seed[s] - mean_rate[tile]))
        cols.append({"tile": tile, "seed": seed, "quantile": q,
                     "mean_rate": mean_rate[tile], "shown_rate": per_seed[seed],
                     "lo": min(per_seed.values()), "hi": max(per_seed.values()),
                     "n_eligible": len(eligible)})
    return cols


def read_rgb(tile: str) -> np.ndarray:
    """True-colour composite, percentile-stretched per chip.

    The .tif is float32 and will not open with PIL; rasterio is required. The stretch is per chip
    because these are display panels, not a radiometric comparison between chips.
    """
    with rasterio.open(REPO / f"data/split_{SPLIT_TAG}/test/images/{tile}.tif") as src:
        a = src.read([1, 2, 3]).astype(np.float32)
    # Per BAND, not over all three at once: a chip that is almost entirely vegetation has a green
    # band far brighter than the other two, and a single stretch turns it into a flat green wash.
    out = np.empty_like(a)
    for b in range(3):
        lo, hi = np.percentile(a[b], [2, 98])
        out[b] = np.clip((a[b] - lo) / (hi - lo), 0, 1)
    return out.transpose(1, 2, 0)


def read_reference(tile: str) -> np.ndarray:
    return np.array(Image.open(REPO / f"data/split_{SPLIT_TAG}/test/masks/{tile}.png").convert("L"))


def read_prediction(tile: str, seed: int) -> np.ndarray:
    p = (REPO / f"analysis/panel_root/seed{seed}/analysis/seed_softmax/{CELL}/seed{seed}/{tile}.npy")
    if not p.is_file():
        raise SystemExit(f"prediction not staged: {p}")
    return np.load(p).argmax(axis=0).astype(np.uint8)


def error_overlay(ref: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """The two pair-error directions over a washed-out reference map."""
    base = PALETTE[ref].astype(np.float32)
    out = (base * (1 - CORRECT_WASH) + 255 * CORRECT_WASH).astype(np.uint8)
    for mask, colour in (((ref == G) & (pred == S), ERR_G2S),
                         ((ref == S) & (pred == G), ERR_S2G)):
        out[mask] = tuple(int(255 * c) for c in mpl.colors.to_rgb(colour))
    return out


def scale_bar(ax, length_m: float = 50.0) -> None:
    n_px = length_m / PX_M
    x0, y = 512 - n_px - 12, 512 - 22
    ax.plot([x0, x0 + n_px], [y, y], color="white", lw=2.6,
            solid_capstyle="butt", clip_on=False, zorder=4)
    ax.plot([x0, x0 + n_px], [y, y], color="black", lw=1.2,
            solid_capstyle="butt", clip_on=False, zorder=5)
    ax.text(x0 + n_px / 2, y - 8, rf"{length_m:.0f}\,m", ha="center", va="bottom", fontsize=7.5,
            color="black", zorder=5,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-path", default="figures/two_grasslands_qualitative.pdf")
    args = ap.parse_args()

    set_plot_style()
    stats = load_tile_stats()
    cols = choose_columns(stats)
    for c in cols:
        # Every scale bar and every hectare in this figure assumes 0.5 m isotropic pixels. The two
        # upland sites are 0.515 x 0.641 m, so a chip from either would silently rescale the bar.
        if not c["tile"].startswith("biodiversity_"):
            raise SystemExit(f"{c['tile']} is not from the inland site; {PX_M} m pixels are wrong")

    rows = ["Image", "Reference", "Prediction", "Pair error"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(FIG_W, 7.95))

    for j, col in enumerate(cols):
        tile, seed = col["tile"], col["seed"]
        ref = read_reference(tile)
        pred = read_prediction(tile, seed)
        pred[ref == 0] = 0
        panels = [read_rgb(tile), PALETTE[ref], PALETTE[pred], error_overlay(ref, pred)]
        for i, img in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(rf"({chr(97+j)})~~{100*col['mean_rate']:.1f}\%", pad=5)
            if j == 0:
                ax.set_ylabel(rows[i], labelpad=5)
        scale_bar(axes[0, j])

    class_handles = [Patch(facecolor=PALETTE[k] / 255, edgecolor="none",
                           label=DISPLAY_NAMES.get(STUDENT_CLASSES[k], STUDENT_CLASSES[k]))
                     for k in range(1, 6)]
    err_handles = [
        Patch(facecolor=ERR_G2S, edgecolor="none", label="Grassland called semi-natural"),
        Patch(facecolor=ERR_S2G, edgecolor="none", label="Semi-natural called grassland"),
    ]
    # Two legends, not one: with seven entries on a single row-wrapped legend matplotlib fills
    # column-major and splits the two error colours away from each other.
    leg1 = fig.legend(handles=class_handles, loc="lower center", ncol=5, frameon=False,
                      bbox_to_anchor=(0.5, 0.036), handlelength=1.2, columnspacing=1.2,
                      handletextpad=0.4)
    fig.add_artist(leg1)
    fig.legend(handles=err_handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.004), handlelength=1.2, columnspacing=1.8,
               handletextpad=0.4)

    fig.subplots_adjust(left=0.055, right=0.995, top=0.955, bottom=0.085,
                        wspace=0.02, hspace=0.02)
    out = REPO / args.out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    # A PNG beside the PDF so the figure can be opened and read without a viewer, which is how
    # every defect on this project has been found. study_area.py does the same.
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)

    print(f"Saved: {out}")
    print(f"  eligible chips: {cols[0]['n_eligible']} of 90")
    for c in cols:
        print(f"  p{c['quantile']:>3d}  {c['tile']}  seed {c['seed']}  "
              f"ten-seed mean {100*c['mean_rate']:5.2f}%  "
              f"(per-seed {100*c['lo']:.2f}-{100*c['hi']:.2f}%), shown {100*c['shown_rate']:5.2f}%")


if __name__ == "__main__":
    main()
