"""Where the foreground error is, by class pair, for the no-intervention cell.

ONE factorial cell, not four. The factorial is a bound on what the design can resolve, not the
paper's finding; several panels of near-identical matrices invited the reader to hunt for
differences between cells that the design cannot separate. Replaces the three-panel
baseline / full / delta figure that shipped as `confusion_matrices.pdf` until 2026-07-29.

Panel (a) is the 5x5 foreground confusion on ABSOLUTE volume -- each off-diagonal cell is that
directed confusion as a share of all foreground error, so the whole panel sums to 100% and cells
are comparable across rows. It is deliberately NOT row-normalised: row normalisation converts the
panel to per-class recall, which hides that one pair holds nearly half the error volume and makes
a rare class's small mistake look the same size as a common class's large one.

Panel (b) puts each unordered pair's observed share beside the share it would hold if confusions
fell out by class size alone (`expected_share_from_area` in narrative_numbers.py). That is the
second half of the claim: the pair is not merely the largest, it is larger than its share of the
ground gives it.

Writes:
  figures/pair_error_confusion.pdf

Run:
  python scripts/figures/pair_error_confusion.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "geoseg").is_dir() and (p / "config").is_dir():
            return p
    raise FileNotFoundError(f"Could not find repo root from {start}")


repo_root = find_repo_root(Path.cwd())
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


# --- WITHDRAWN-DATA GUARD, added 2026-07-27 ---------------------------------------------------
# analysis/eval_219/ is the WITHDRAWN campaign's output: 219 val tiles of the random-by-tile split,
# whose held-out tiles share ~93% of their ground with training. This script builds a MANUSCRIPT
# figure, so without this it rebuilds a paper figure from withdrawn numbers and exits 0.
def _refuse_withdrawn(p):
    if "eval_219" in Path(p).parts:
        raise SystemExit(
            f"{p}\n  is the WITHDRAWN pre-2026-07-26 campaign (leaking 219-tile random split).\n"
            f"  This figure has not been re-pointed at the current campaign's output.")
# ---------------------------------------------------------------------------------------------

SPLIT = os.environ.get("FIG_SPLIT", "test")
if SPLIT not in ("test", "external_test"):
    raise SystemExit(f"FIG_SPLIT={SPLIT!r}; expected 'test' or 'external_test'")
EVAL_DIR = repo_root / "analysis/confusion"
_refuse_withdrawn(EVAL_DIR)
CELL = "stage1_baseline"

OUT_DIR = repo_root / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# SPLIT is in the filename so a Test B build cannot overwrite the Test A figure and be published
# under a Test A caption. FIG_SPLIT is whitelisted for the same reason.
OUT_PDF = OUT_DIR / ("pair_error_confusion.pdf" if SPLIT == "test"
                     else f"pair_error_confusion_{SPLIT}.pdf")

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{lmodern}",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

CLASS_NAMES_FG = ["Forest", "Grassland", "Cropland", "Settlement", "Semi-nat."]
PAIR_LABELS = {
    "Grassland-Seminatural": "Grassland -- Semi-nat.",
    "Forest-Grassland": "Forest -- Grassland",
    "Grassland-Cropland": "Grassland -- Cropland",
    "Grassland-Settlement": "Grassland -- Settlement",
    "Forest-Seminatural": "Forest -- Semi-nat.",
    "Forest-Settlement": "Forest -- Settlement",
    "Cropland-Settlement": "Cropland -- Settlement",
    "Forest-Cropland": "Forest -- Cropland",
    "Settlement-Seminatural": "Settlement -- Semi-nat.",
    "Cropland-Seminatural": "Cropland -- Semi-nat.",
}

# PAGE WIDTH. \textwidth in Definitions/mdpi.cls is 394.36 pt = 13.90 cm; \extralength is 4.61 cm,
# so a figure inside \begin{adjustwidth}{-\extralength}{0cm} -- the pattern main.tex already uses --
# gets 18.51 cm = 7.28 in. Built at that width, 1 pt here is 1 pt on the page; built any wider it is
# scaled down by \includegraphics[width=\linewidth] and every label shrinks with it.
FIG_W = 7.28
CMAP = "Blues"
ACCENT = "#3C6E9E"      # the study-area figure's inland-site blue
MUTED = "#9AA5AE"
DIAG_GREY = "#E8E8E8"


def load_confusion():
    cm_path = EVAL_DIR / f"confusion_{SPLIT}_{CELL}.npy"
    if not cm_path.exists():
        raise FileNotFoundError(
            f"{cm_path} missing -- run scripts/analysis/pooled_confusion.py "
            f"--split {SPLIT} --cell {CELL}")
    meta = json.loads((EVAL_DIR / f"confusion_{SPLIT}_{CELL}.json").read_text())
    return np.load(cm_path).astype(np.int64), meta


def error_share_matrix(cm: np.ndarray) -> np.ndarray:
    """Off-diagonal foreground confusion as a share of ALL foreground error.

    Rows are the reference class, columns the prediction, background dropped from both. The single
    denominator is what makes cells comparable ACROSS rows; row normalisation would not be.
    """
    fg = cm[1:, 1:].astype(np.float64)
    total_err = cm[1:, :].sum() - np.trace(cm[1:, 1:])
    out = fg / total_err
    np.fill_diagonal(out, np.nan)
    return out


def plot_matrix(ax, share):
    masked = np.ma.masked_invalid(share)
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad(DIAG_GREY)
    im = ax.imshow(masked, interpolation="nearest", vmin=0.0, vmax=np.nanmax(share), cmap=cmap)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(CLASS_NAMES_FG, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES_FG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    hi = 0.45 * np.nanmax(share)
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            v = 100 * share[i, j]
            ax.text(j, i, f"{v:.1f}" if v >= 0.05 else r"$<$0.1",
                    ha="center", va="center", fontsize=7.5,
                    color="white" if share[i, j] > hi else "black")
    return im


def plot_pairs(ax, pairs):
    pairs = sorted(pairs, key=lambda d: d["share_of_error"])
    y = np.arange(len(pairs))
    obs = np.array([100 * d["share_of_error"] for d in pairs])
    exp = np.array([100 * d["expected_share_from_area"] for d in pairs])

    for yi, o, e in zip(y, obs, exp):
        ax.plot([e, o], [yi, yi], color=MUTED, lw=1.0, zorder=1)
    ax.scatter(exp, y, s=26, facecolor="white", edgecolor=MUTED, lw=1.0, zorder=2)
    ax.scatter(obs, y, s=26, color=ACCENT, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([PAIR_LABELS[d["pair"]] for d in pairs])
    ax.set_xlabel(r"Share of foreground error (\%)")
    ax.set_xlim(-2, 58)
    ax.set_ylim(-1.6, len(pairs) - 0.35)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    top = pairs[-1]
    ax.annotate(f"{100 * top['share_of_error']:.1f}\\%",
                xy=(100 * top["share_of_error"], y[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=8, color=ACCENT)
    ax.annotate(f"{100 * top['expected_share_from_area']:.1f}\\%",
                xy=(100 * top["expected_share_from_area"], y[-1]), xytext=(-6, 0),
                textcoords="offset points", va="center", ha="right", fontsize=8, color=MUTED)
    # A two-entry legend, not words placed over the top row's markers. On the row below, the open
    # and filled markers swap sides, so a positional label misreads Forest--Grassland backwards.
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor=MUTED, markersize=5, label="expected from class area"),
        Line2D([], [], marker="o", linestyle="none", color=ACCENT, markersize=5,
               label="observed"),
    ], loc="lower right", frameon=False, fontsize=8, handletextpad=0.4,
        borderaxespad=0.2, labelspacing=0.3)


def main():
    cm, meta = load_confusion()
    share = error_share_matrix(cm)
    pairs = json.loads(
        (repo_root / "artifacts/narrative_numbers_test.json").read_text())["pair_ratios"]

    n_seeds = len(meta["seeds"])
    err_ha_per_seed = meta["foreground_errors"] / n_seeds * 0.25 / 1e4

    fig = plt.figure(figsize=(FIG_W, 3.05), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.02], wspace=0.62)

    ax1 = fig.add_subplot(gs[0, 0])
    # No colourbar. Every cell carries its number, so a scale bar would restate them and, at this
    # width, its ticks landed on top of panel (b)'s pair labels. The shading is there to make the
    # two dark cells findable at a glance, not to be read off.
    plot_matrix(ax1, share)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_pairs(ax2, pairs)

    # Panel titles as figure text at one shared height, not axes titles. (a) has equal aspect, so
    # its axes box is shorter than (b)'s and axes titles landed at two different heights.
    for ax, text in ((ax1, r"(a) Share of all foreground error (\%)"),
                     (ax2, r"(b) Against what class area alone gives")):
        box = ax.get_position()
        fig.text(box.x0 + box.width / 2, 0.965, text, ha="center", va="top", fontsize=9)

    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    # A PNG beside the PDF so the figure can be opened and read without a viewer, which is how
    # every defect on this project has been found. study_area.py does the same.
    fig.savefig(OUT_PDF.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PDF}")
    print(f"  {CELL}, {SPLIT}, {meta['n_tiles']} chips, {n_seeds} seeds; "
          f"{meta['foreground_errors']:,} foreground error px "
          f"({err_ha_per_seed:.1f} ha per seed)")
    gs_pair = next(d for d in pairs if d["pair"] == "Grassland-Seminatural")
    print(f"  Grassland--Seminatural: {100 * gs_pair['share_of_error']:.2f}% of foreground error, "
          f"{gs_pair['ratio']:.3f}x the share class area alone gives it")


if __name__ == "__main__":
    main()
