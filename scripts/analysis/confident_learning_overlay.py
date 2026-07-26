"""Two-method spatial overlay: our ensemble entropy vs confident-learning flags.

Shows, on the same validation tiles, that an independent label-error detector
(confident learning; Northcutt et al. 2021) flags the same boundary pixels our
ten-seed ensemble entropy lights up. Reuses the signature overlay style of
draft_boundary_overlay.py (Computer Modern, magma entropy, white-over-black GT
class-boundary contours) so it sits beside the uncertainty-overlay figure.

Columns: (a) RGB, (b) ground truth, (c) ensemble entropy + GT boundaries,
(d) confident-learning label-issue flags + GT boundaries.

Confident learning is run on ALL foreground val pixels (background excluded,
probabilities renormalised over the five foreground classes), then the flags are
scattered back to the two illustrated tiles. Run (label-quality-ceiling env):
    python scripts/analysis/confident_learning_overlay.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend import Legend

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geoseg.taxonomy import STUDENT_CLASSES  # noqa: E402
C = len(STUDENT_CLASSES)
from scripts.analysis.confident_learning_check import build_arrays  # noqa: E402

import os  # noqa: E402
# The untagged split directories belong to the WITHDRAWN random split (219 val / 218 test tiles).
SPLIT_TAG = os.environ.get("SPLIT_TAG", "f1")

from scripts.analysis.draft_boundary_overlay import (  # noqa: E402
    setup_font, gt_boundary_mask, _rgb_for, PAL, CLASS_CMAP, CLASS_NORM,
)

DEFAULT_TILES = ["biodiversity_1969", "biodiversity_2126"]   # same as the uncertainty overlay
FLAG_RGB = np.array([0.10, 0.83, 0.78])     # turquoise = flagged label issue (no cyan in magma)
INSIDE_RGB = np.array([0.12, 0.12, 0.14])   # near-black annotated interior
OUTSIDE_RGB = np.array([1.0, 1.0, 1.0])     # white outside the annotated farm extent


def cleanlab_flag_maps(softmax_root, mask_dir, cell, seeds):
    """Run foreground confident learning on the whole val set; return per-tile flag maps."""
    ids, labels, pred, ent, dist = build_arrays(softmax_root, mask_dir, cell, seeds)
    fg = np.isin(labels, np.arange(1, C))
    probs = np.transpose(pred, (0, 2, 3, 1))[fg][:, 1:C]
    probs = (probs / probs.sum(axis=1, keepdims=True)).astype(np.float32)
    y = (labels[fg] - 1).astype(np.int64)

    from cleanlab.filter import find_label_issues
    print("[cleanlab] confident learning on foreground val pixels...")
    issues_flat = find_label_issues(y, probs, filter_by="both", n_jobs=1)
    flag_map = np.zeros(labels.shape, dtype=bool)
    flag_map[fg] = issues_flat
    return {iid: {"mask": labels[i], "pred": pred[i].argmax(axis=0).astype(np.uint8),
                  "ent": ent[i], "flags": flag_map[i]}
            for i, iid in enumerate(ids)}


def render(tiles, data, img_dir, out_dir, use_tex):
    setup_font(use_tex)
    n = len(tiles)
    # Self-contained four panels: GT and prediction give the reader the class context, then our
    # entropy and the confident-learning flags. No GT-boundary contours: the GT panel already
    # carries the class regions, and contours would occlude the signal that sits on the boundaries.
    panels = [{"iid": iid, "mask": data[iid]["mask"], "pred": data[iid]["pred"],
               "ent": data[iid]["ent"], "flags": data[iid]["flags"]} for iid in tiles]
    vmax = float(np.ceil(max(p["ent"].max() for p in panels) * 20) / 20)

    # Geometry mirrors the uncertainty-overlay figure (draft_boundary_overlay): same per-panel size
    # (~2.64 in) and the same setup_font defaults, so panels and text match when this is included at
    # 0.72*\fulllength in LaTeX (that figure is \fulllength wide with ~5.76 width-ratio units).
    fig = plt.figure(figsize=(10.9, 2.62 * n))
    # The entropy colourbar sits BETWEEN (c) Entropy and (d) CL flags, so the scale is adjacent to
    # the entropy panel it describes rather than stranded to the right of (d). Columns:
    # 0=(a) 1=(b) 2=(c) 3=colourbar 4=(d) 5=right spacer.
    # col 3 = entropy colourbar (adjacent to (c) Entropy), col 4 = spacer reserving room for the
    # colourbar's tick labels + axis label, so they don't collide with (d).
    gs = fig.add_gridspec(n, 7, width_ratios=[1, 1, 1, 0.10, 0.30, 1, 0.03],
                          wspace=0.05, hspace=0.05,
                          left=0.008, right=0.985, top=0.90, bottom=0.21)
    axes = np.empty((n, 4), dtype=object)
    panel_cols = [0, 1, 2, 5]   # (a) GT, (b) Prediction, (c) Entropy, (d) CL flags
    for r in range(n):
        for j, c in enumerate(panel_cols):
            axes[r, j] = fig.add_subplot(gs[r, c])
    cax_e = fig.add_subplot(gs[:, 3])

    im_ent = None
    for r, p in enumerate(panels):
        axes[r, 0].imshow(p["mask"], cmap=CLASS_CMAP, norm=CLASS_NORM, interpolation="nearest")
        axes[r, 1].imshow(p["pred"], cmap=CLASS_CMAP, norm=CLASS_NORM, interpolation="nearest")
        im_ent = axes[r, 2].imshow(p["ent"], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        # confident-learning flags: turquoise on a near-black annotated interior, white outside
        canvas = np.ones((*p["flags"].shape, 3)) * OUTSIDE_RGB
        canvas[p["mask"] != 0] = INSIDE_RGB
        canvas[p["flags"]] = FLAG_RGB
        axes[r, 3].imshow(canvas, interpolation="nearest")
        if r == 0:
            axes[r, 0].set_title("(a) Ground truth")
            axes[r, 1].set_title("(b) Prediction")
            axes[r, 2].set_title("(c) Entropy")
            axes[r, 3].set_title("(d) CL flags")
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])

    cb = fig.colorbar(im_ent, cax=cax_e)
    cb.set_label(r"$H[\bar{p}]\ /\ \log 6$" if use_tex else "entropy / log 6", fontsize=17)
    cb.ax.tick_params(labelsize=15)

    # Legend mirrors the uncertainty-overlay figure's structure (draft_boundary_overlay.add_legend):
    # the five class colours label the class-coloured map panels (a)--(c), so their key sits under
    # that block split 3 + 2 (Forest/Grassland/Cropland over Settlement/Semi-natural), each row
    # centred; "Flagged label issue" appears only in panel (d), so its key sits under (d).
    class_handles = [Patch(facecolor=PAL[k], edgecolor="0.3",
                           label=STUDENT_CLASSES[k].replace("Seminatural", "Semi-natural"))
                     for k in range(1, C)]
    flag_handle = Patch(facecolor=FLAG_RGB, edgecolor="0.3", label="Flagged label issue")

    def add_legend(handles, ncol, x, y):
        leg = Legend(fig, handles, [h.get_label() for h in handles], loc="lower center",
                     ncol=ncol, bbox_to_anchor=(x, y), bbox_transform=fig.transFigure,
                     frameon=False, fontsize=18, columnspacing=1.3,
                     handlelength=1.5, handletextpad=0.5)
        fig.add_artist(leg)

    # Centre on the ACTUAL panel positions (get_position accounts for wspace): the class legend
    # sits under the label maps (a)-(b) -- the only panels that use the class colours -- and the
    # flag key under (d) CL flags, the only panel that uses the flag colour.
    p_a = axes[0, 0].get_position(); p_b = axes[0, 1].get_position(); p_d = axes[0, 3].get_position()
    x_ab = 0.5 * (p_a.x0 + p_b.x1)
    x_d = 0.5 * (p_d.x0 + p_d.x1)
    add_legend(class_handles[:3], 3, x_ab, 0.065)    # Forest, Grassland, Cropland under (a)-(b)
    add_legend(class_handles[3:], 2, x_ab, 0.000)    # Settlement, Semi-natural, centred beneath
    add_legend([flag_handle], 1, x_d, 0.033)         # Flagged label issue under (d)

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "confident_learning_overlay.pdf"
    png = out_dir / "confident_learning_overlay.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.15, dpi=300)
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return pdf, png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--softmax-root", default="sonic/results")
    ap.add_argument("--mask-dir", default=f"data/split_{SPLIT_TAG}/val/masks")
    ap.add_argument("--img-dir", default=f"data/split_{SPLIT_TAG}/val/images")
    ap.add_argument("--cell", default="stage3_clsbal")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--tiles", nargs="+", default=DEFAULT_TILES)
    ap.add_argument("--out-dir", default="analysis/label_ceiling")
    ap.add_argument("--fig-dir", default="manuscript/Figures")
    ap.add_argument("--no-tex", action="store_true")
    args = ap.parse_args()

    data = cleanlab_flag_maps(args.softmax_root, args.mask_dir, args.cell, args.seeds)
    for iid in args.tiles:
        fr = float(data[iid]["flags"][data[iid]["mask"] != 0].mean()) * 100
        print(f"  {iid}: {fr:.2f}% of annotated pixels flagged")
    pdf, png = render(args.tiles, data, args.img_dir, args.out_dir, not args.no_tex)
    # also drop the PDF into the manuscript Figures dir
    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    dest = fig_dir / "confident_learning_overlay.pdf"
    dest.write_bytes(Path(pdf).read_bytes())
    print(f"[written] {pdf}\n[written] {png}\n[written] {dest}")


if __name__ == "__main__":
    main()
