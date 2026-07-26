#!/usr/bin/env python3
"""
Export a LaTeX tabular snippet summarising test-set performance.

Reads:
  evaluation/evaluation_results/test/stage1_baseline/metrics.json
  evaluation/evaluation_results/test/stage3_clsbal/metrics.json

Writes:
  evaluation/evaluation_results/final_test_table.tex

The output is a bare tabularx environment (no \\begin{table} float) so it can be
\\input{} inside any table/figure environment in the manuscript.

Usage (from repo root):
  python evaluation/export_final_test_table.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path


# ── Paths (relative to repo root) ──────────────────────────────────────────

SPLIT_TAG = os.environ.get("SPLIT_TAG", "f1")
# Repo-root-relative, not CWD-relative: run from anywhere but the repo root, the old paths either
# errored or wrote the manuscript table into whatever tree the shell happened to be in.
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "artifacts").is_dir())
TEST_DIR = REPO_ROOT / "evaluation/evaluation_results/test"
OUTPUT_PATH = REPO_ROOT / "evaluation/evaluation_results/final_test_table.tex"

# Stages to include (order matters — rows appear in this order)
STAGES = [
    ("stage1_baseline", "Stage 1 (baseline)"),
    ("stage3_clsbal", "Stage 3 (clsbal)"),
]

# Class names matching compute_metrics.py (foreground only)
CLASS_NAMES_5 = ["Forest land", "Grassland", "Cropland", "Settlement", "Seminatural Grassland"]

# Display names matching Table 2 in the manuscript
DISPLAY_NAMES = ["Forest", "Grassland", "Cropland", "Settlement", "Semi-natural"]


def load_metrics(stage_dir: str) -> dict:
    # compute_metrics.py names the run directory after the checkpoint's parent, which SPLIT_TAG
    # tags; spelling the cell untagged here resolved to the withdrawn campaign's directory.
    path = TEST_DIR / f"{stage_dir}_{SPLIT_TAG}" / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {path}\n"
            "Run evaluation/compute_metrics.py --split test first."
        )
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    # Resolving by tag is not enough on its own: a metrics.json can be copied, restored from a
    # fetch, or left behind by an earlier campaign at a path that now looks current. Every one
    # records the checkpoint that produced it, so check it rather than trust the directory name.
    got = Path(m.get("checkpoint", "")).name
    want = f"{stage_dir}_{SPLIT_TAG}.ckpt"
    if got != want:
        raise SystemExit(
            f"{path}\n  was produced by '{got or '(none recorded)'}', not '{want}'. This is a "
            f"metrics file from another cell, another split tag, or the withdrawn campaign.")
    if m.get("tta") is not False:
        raise SystemExit(f"{path}\n  has tta={m.get('tta')}; the reported table is the no-TTA run.")
    return m


def main() -> None:
    # Load all stage metrics
    all_metrics = []
    for stage_dir, label in STAGES:
        m = load_metrics(stage_dir)
        all_metrics.append((label, m))

    # Build LaTeX tabularx (bare — no \begin{table} wrapper)
    lines = [
        r"\begin{tabularx}{\textwidth}{",
        r"    >{\raggedright\arraybackslash}p{2.4cm}",
        r"    >{\centering\arraybackslash}p{1.3cm}",
        r"    >{\centering\arraybackslash}p{1.3cm}",
        r"    >{\centering\arraybackslash}p{1.3cm}",
        r"    >{\centering\arraybackslash}p{1.6cm}",
        r"    >{\centering\arraybackslash}p{1.6cm}",
        r"    >{\centering\arraybackslash}p{1.0cm}",
        r"    >{\centering\arraybackslash}p{1.0cm}",
        r"    >{\centering\arraybackslash}X",
        r"}",
        r"\toprule",
        r"\begin{tabular}[c]{@{}c@{}} \textbf{Stage} \end{tabular} &",
        r"\begin{tabular}[c]{@{}c@{}} \textbf{Forest} \\ (\%) \end{tabular} &",
        r"\begin{tabular}[c]{@{}c@{}} \textbf{Grassland} \\ (\%) \end{tabular} &",
        r"\begin{tabular}[c]{@{}c@{}} \textbf{Cropland} \\ (\%) \end{tabular} &",
        r"\begin{tabular}[c]{@{}c@{}} \textbf{Settlement} \\ (\%) \end{tabular} &",
        r"\begin{tabular}[c]{@{}c@{}} \textbf{Semi-natural} \\ (\%) \end{tabular} &",
        r"\begin{tabular}[c]{@{}c@{}} \textbf{mIoU} \\ (\%) \end{tabular} &",
        r"\begin{tabular}[c]{@{}c@{}} \textbf{mF1} \\ (\%) \end{tabular} &",
        r"\begin{tabular}[c]{@{}c@{}} \textbf{OA} \\ (\%) \end{tabular}",
        r"\\",
        r"\midrule",
    ]

    # Data rows — 1 decimal place to match Tables 1–2
    for label, m in all_metrics:
        oa = m["OA"] * 100
        miou = m["mIoU_excluding_bg"] * 100
        mf1 = m["mF1_excluding_bg"] * 100
        iou_vals = [m["per_class_iou"][c] * 100 for c in CLASS_NAMES_5]
        data_cells = f"{label} & " + " & ".join(f"{v:.1f}" for v in iou_vals)
        data_cells += f" & {miou:.1f} & {mf1:.1f} & {oa:.1f}"
        lines.append(data_cells + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")

    # Write
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Summary to stdout
    print(f"Output: {OUTPUT_PATH}")
    print()
    for label, m in all_metrics:
        miou = m["mIoU_excluding_bg"] * 100
        mf1 = m["mF1_excluding_bg"] * 100
        oa = m["OA"] * 100
        per_class_iou = {c: m["per_class_iou"][c] * 100 for c in CLASS_NAMES_5}
        print(f"Test-set metrics ({label}):")
        for name, key in zip(DISPLAY_NAMES, CLASS_NAMES_5):
            print(f"  {name:14s} IoU = {per_class_iou[key]:.1f}%")
        print(f"  {'mIoU':14s}     = {miou:.1f}%")
        print(f"  {'mF1':14s}     = {mf1:.1f}%")
        print(f"  {'OA':14s}     = {oa:.1f}%")
        print()


if __name__ == "__main__":
    main()
