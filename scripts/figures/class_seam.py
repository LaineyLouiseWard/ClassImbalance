#!/usr/bin/env python3
"""
Do the two classes that carry the error actually meet on the ground?

THE ARGUMENT THIS FIGURE HAS TO CARRY. Grassland and semi-natural grassland account for 46.7% of
foreground error. If that error were a seam being drawn in the wrong place, the two classes would
have to have a seam. They barely do: 0.60% of grassland's ground lies within 8 m of any
semi-natural, against 21.44% within 8 m of forest. The pair carrying half the error is the pair
that almost never touches.

Everything here is arithmetic on the two REFERENCE masks. No model output enters it, so it is a
bound on what any model could be doing, not a property of this one.

Panel (a) is the whole matrix: for each row class, the share of its ground lying within 8 m of each
column class, over the 90 Test A scoring chips. Panel (b) shows one chip so a reader can see what
0.6% looks like beside 21%.

CHIP SELECTION for panel (b), stated so the caption can repeat it: of the 90 chips, those that are
fully labelled and contain grassland, semi-natural AND forest, the one minimising the larger of the
two absolute log deviations from the pooled pair of values (0.60% near semi-natural, 21.44% near
forest). That is the chip closest to typical on BOTH numbers at once, in relative terms. Choosing on
the semi-natural number alone returns a chip whose forest seam is twice the pooled value, which
would overstate the contrast; the two-coordinate rule is what stops the illustration flattering the
claim.

Writes:
  figures/class_seam.pdf  (+ .png)

Run:
  python scripts/figures/class_seam.py
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

from geoseg.geo import gsd_for  # noqa: E402
from geoseg.taxonomy import STUDENT_PALETTE  # noqa: E402

SPLIT_TAG = "f1"
NEAR_M = 8.0
FOREST, GRASS, SEMI = 1, 2, 5
FG_ORDER = ["Forest", "Grassland", "Cropland", "Settlement", "Seminatural"]
SHORT = ["Forest", "Grassland", "Cropland", "Settlement", "Semi-nat."]

# PAGE WIDTH. \textwidth in Definitions/mdpi.cls is 394.36 pt = 13.90 cm; \extralength is 4.61 cm,
# so a figure inside \begin{adjustwidth}{-\extralength}{0cm} -- the pattern main.tex already uses --
# gets 18.51 cm = 7.28 in. Built at that width, 1 pt here is 1 pt on the page; built any wider it is
# scaled down by \includegraphics[width=\linewidth] and every label shrinks with it.
FIG_W = 7.28
PALETTE = np.array(STUDENT_PALETTE, dtype=np.uint8)
BAND_COLOUR = "#3B2A6B"     # the 8 m band; one colour, used for both bands so they compare directly
WASH = 0.55                 # light enough that the semi-natural and forest the band is measured
                            # AGAINST stay visible; a heavier wash hid them and left panel (c)
                            # looking like an empty chip rather than one with no seam


def set_plot_style() -> None:
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{lmodern}",
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


def near_matrix() -> np.ndarray:
    """Pooled shares from artifacts/class_adjacency_test.json, as a 5x5 with a NaN diagonal."""
    d = json.loads((REPO / "artifacts/class_adjacency_test.json").read_text())
    m = np.full((5, 5), np.nan)
    for r in d["pairs"]:
        m[FG_ORDER.index(r["a"]), FG_ORDER.index(r["b"])] = 100 * r["share_of_a_within_8m_of_b"]
    if np.isnan(m[~np.eye(5, dtype=bool)]).any():
        raise SystemExit("class_adjacency_test.json is missing ordered pairs; rerun class_adjacency.py")
    return m


def chip_shares(tile: str):
    """(share of grassland within 8 m of semi-natural, same for forest) on one chip, plus the mask."""
    m = np.array(Image.open(REPO / f"data/split_{SPLIT_TAG}/test/masks/{tile}.png").convert("L"))
    sampling = gsd_for(tile)
    g = m == GRASS
    if not g.any():
        return None
    d_semi = ndimage.distance_transform_edt(m != SEMI, sampling=sampling)
    d_forest = ndimage.distance_transform_edt(m != FOREST, sampling=sampling)
    band_s = g & (d_semi < NEAR_M)
    band_f = g & (d_forest < NEAR_M)
    return (100 * band_s.sum() / g.sum(), 100 * band_f.sum() / g.sum(), m, band_s, band_f)


def choose_chip(pooled_semi: float, pooled_forest: float):
    subset = json.loads((REPO / f"artifacts/scoring_subset_{SPLIT_TAG}.json").read_text())
    tiles = sorted(subset["splits"]["test"]["tiles"])
    best = None
    n_eligible = 0
    for t in tiles:
        m = np.array(Image.open(REPO / f"data/split_{SPLIT_TAG}/test/masks/{t}.png").convert("L"))
        if (m == 0).any():
            continue
        if not ((m == GRASS).any() and (m == SEMI).any() and (m == FOREST).any()):
            continue
        n_eligible += 1
        xs, xf, mask, band_s, band_f = chip_shares(t)
        # Log deviation, so "twice the pooled value" and "half of it" are penalised equally. A plain
        # difference would be dominated by the forest coordinate, which is 36x larger.
        if xs <= 0 or xf <= 0:
            # A chip with no grassland at all within 8 m of one of the two classes is infinitely
            # far from the pooled value in log terms, so it cannot be the closest. Skipping it
            # explicitly rather than letting log(0) do it keeps the intent readable.
            continue
        dev = max(abs(np.log(xs / pooled_semi)), abs(np.log(xf / pooled_forest)))
        if best is None or dev < best["dev"]:
            best = {"tile": t, "dev": dev, "near_semi": xs, "near_forest": xf,
                    "mask": mask, "band_s": band_s, "band_f": band_f}
    if best is None:
        raise SystemExit("no eligible chip for the seam illustration")
    best["n_eligible"] = n_eligible
    return best


def plot_matrix(ax, m):
    cmap = plt.get_cmap("Purples").copy()
    cmap.set_bad("#E8E8E8")
    # PowerNorm, not linear. On a linear 0-70 scale the two largest cells both sit at max-dark and
    # everything below about 25 collapses into one near-white band -- which is where 0.6, 6.1 and
    # 21.4 live, the three cells the argument turns on. gamma=0.5 spreads the low end without
    # capping anything, so the ordering stays monotone and no value is thrown away.
    norm = mpl.colors.PowerNorm(gamma=0.5, vmin=0, vmax=np.nanmax(m))
    ax.imshow(np.ma.masked_invalid(m), cmap=cmap, norm=norm)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(SHORT, rotation=45, ha="right")
    ax.set_yticklabels(SHORT)
    ax.set_xlabel(r"\ldots{} lying within 8\,m of this class")
    ax.set_ylabel(r"Share of this class's ground \ldots{}")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            # Switch on the NORMALISED value, not the raw one: with a non-linear norm the raw value
            # no longer tells you how dark the cell is, which is what the text has to contrast with.
            ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if norm(m[i, j]) > 0.6 else "black")
    # The two cells of the pair that carries 46.7% of the error, in both directions. An outline,
    # not a colour: the cells already carry a colour that means something else. The comparison the
    # argument rests on is inside the outlined row -- 0.6 against 21.4 -- so nothing else is marked.
    # Inset, and clipping off. imshow spans -0.5 to 4.5, so a box drawn on the cell boundary in
    # the last row or column has half its stroke outside the axes and that edge is clipped -- the
    # outline then renders thinner on two sides than the other two.
    inset = 0.055
    for (i, j) in ((1, 4), (4, 1)):
        ax.add_patch(plt.Rectangle((j - 0.5 + inset, i - 0.5 + inset),
                                   1 - 2 * inset, 1 - 2 * inset, fill=False,
                                   edgecolor="#B2182B", lw=1.8, joinstyle="miter",
                                   clip_on=False, zorder=5))


def plot_chip(axes, chip):
    mask = chip["mask"]
    base = (PALETTE[mask].astype(np.float32) * (1 - WASH) + 255 * WASH).astype(np.uint8)
    rgb = np.array(mpl.colors.to_rgb(BAND_COLOUR)) * 255
    for ax, band, title in (
            (axes[0], chip["band_f"],
             "grassland within\n"
             rf"8\,m of forest: {chip['near_forest']:.1f}\%"),
            (axes[1], chip["band_s"],
             "grassland within\n"
             rf"8\,m of semi-natural: {chip['near_semi']:.2f}\%")):
        img = base.copy()
        img[band] = rgb.astype(np.uint8)
        ax.imshow(img)
        ax.set_title(title, fontsize=8, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        # The panel is about an 8 m distance, so the reader needs to see how big 8 m is on it.
        n_px = 50.0 / 0.5
        x0, ybar = 512 - n_px - 12, 512 - 20
        ax.plot([x0, x0 + n_px], [ybar, ybar], color="black", lw=1.2, solid_capstyle="butt",
                zorder=5, path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])
        ax.text(x0 + n_px / 2, ybar - 8, r"50\,m", ha="center", va="bottom", fontsize=7,
                zorder=5, path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-path", default="figures/class_seam.pdf")
    args = ap.parse_args()

    set_plot_style()
    m = near_matrix()
    pooled_semi = m[FG_ORDER.index("Grassland"), FG_ORDER.index("Seminatural")]
    pooled_forest = m[FG_ORDER.index("Grassland"), FG_ORDER.index("Forest")]
    chip = choose_chip(pooled_semi, pooled_forest)

    fig = plt.figure(figsize=(FIG_W, 3.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.22, 1.0, 1.0], wspace=0.14)
    ax_m = fig.add_subplot(gs[0, 0])
    plot_matrix(ax_m, m)
    ax_m.set_title("")

    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    plot_chip((ax_b, ax_c), chip)
    titles = [(ax_m, r"(a) All 90 Test A chips"),
              (ax_b, "(b) " + ax_b.get_title()),
              (ax_c, "(c) " + ax_c.get_title())]
    for ax, _ in titles:
        ax.set_title("")
    fig.canvas.draw()
    # One shared height for all three, in figure coordinates. Panel (a) has equal aspect so its
    # axes box is shorter than the chip panels', and axes titles therefore sat at three different
    # heights. Taken from the tallest box so nothing overlaps.
    ax_a, text_a = titles[0]
    box_a = ax_a.get_position()
    t_a = fig.text(box_a.x0 + box_a.width / 2, box_a.get_points()[1][1] + 0.030, text_a,
                   ha="center", va="bottom", fontsize=9, linespacing=1.25)
    # Read (a)'s rendered top back rather than assuming a line height, then hang the two-line chip
    # titles from it. They then start level with (a) and end just above their own images.
    fig.canvas.draw()
    top_a = t_a.get_window_extent().transformed(fig.transFigure.inverted()).y1
    for ax, text in titles[1:]:
        box = ax.get_position()
        fig.text(box.x0 + box.width / 2, top_a, text,
                 ha="center", va="top", fontsize=9, linespacing=1.25)
    # The chip id belongs in the caption, not stamped on the figure -- it is provenance, not
    # something a reader reads off the panels.

    out = REPO / args.out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.03, dpi=300)
    plt.close(fig)

    # Printed width of the matrix, as a fraction of the tight-cropped figure. Both matrix
    # figures are placed at the same \\linewidth, so equal fractions mean equal printed size.
    _bb = fig.get_tightbbox(fig.canvas.get_renderer())
    _ax = ax_m.get_window_extent()
    print(f"  matrix width: {100 * (_ax.width / fig.dpi) / _bb.width:.2f}% of the cropped figure")
    print(f"Saved: {out}")
    print(f"  pooled: grassland within 8 m of semi-natural {pooled_semi:.2f}%, "
          f"of forest {pooled_forest:.2f}%")
    print(f"  illustration chip {chip['tile']} chosen from {chip['n_eligible']} eligible: "
          f"{chip['near_semi']:.2f}% / {chip['near_forest']:.2f}% "
          f"(max log deviation {chip['dev']:.3f})")


if __name__ == "__main__":
    main()
