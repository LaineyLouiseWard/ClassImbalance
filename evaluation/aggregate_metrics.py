#!/usr/bin/env python3
"""
Aggregate metrics.json files produced by compute_metrics.py.

Reads:
  evaluation/evaluation_results/**/metrics.json

Writes:
  evaluation/evaluation_results/metrics_summary.txt
"""

import json
import os
from pathlib import Path
import numpy as np


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Aggregate metrics.json files from evaluation results")
    ap.add_argument(
        "--eval-root",
        type=str,
        default="evaluation/evaluation_results",
        help="Root folder created by compute_metrics.py",
    )
    ap.add_argument(
        "--out-file",
        type=str,
        default="evaluation/evaluation_results/metrics_summary.txt",
        help="Output summary text file",
    )
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    # Only this split tag's run directories. An unfiltered rglob picks up the WITHDRAWN campaign's
    # untagged directories, which are still on disk under evaluation/evaluation_results/val/, and
    # ranks them in the same table as the current cells -- plausible numbers off a leaking split,
    # exit 0. Matching on the directory name is the same rule compute_metrics.py writes under.
    split_tag = os.environ.get("SPLIT_TAG", "f1")
    metrics_files = sorted(p for p in eval_root.rglob("metrics.json")
                           if p.parent.name.endswith(f"_{split_tag}"))
    if not metrics_files:
        found = sorted(p.parent.name for p in eval_root.rglob("metrics.json"))
        raise FileNotFoundError(
            f"No metrics.json for SPLIT_TAG={split_tag} under {eval_root}. Run compute_metrics.py "
            f"first, or set SPLIT_TAG to the campaign you mean. Run directories present: "
            f"{found or '(none)'}"
        )

    rows = []
    for mf in metrics_files:
        with open(mf, "r", encoding="utf-8") as f:
            m = json.load(f)
        rows.append(
            {
                "run_dir": str(mf.parent),
                "checkpoint": Path(m.get("checkpoint", "")).name,
                "OA": float(m.get("OA", np.nan)),
                "mIoU_excluding_bg": float(m.get("mIoU_excluding_bg", np.nan)),
                "mF1_excluding_bg": float(m.get("mF1_excluding_bg", np.nan)),
            }
        )

    def write_block(fh, key: str) -> None:
        fh.write(f"\n{key} (sorted desc)\n")
        fh.write("=" * 70 + "\n")
        sorted_rows = sorted(
            rows,
            key=lambda r: (r[key] if not np.isnan(r[key]) else -1),
            reverse=True,
        )
        for r in sorted_rows:
            fh.write(f"{r[key]:.4f}  | {r['checkpoint']}\n")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(f"Found {len(rows)} evaluated checkpoints under: {eval_root}\n")
        write_block(fh, "mIoU_excluding_bg")
        write_block(fh, "mF1_excluding_bg")
        write_block(fh, "OA")

        fh.write("\n\nFull listing:\n")
        fh.write("=" * 70 + "\n")
        for r in rows:
            fh.write(
                f"{r['mIoU_excluding_bg']:.4f} mIoU | "
                f"{r['mF1_excluding_bg']:.4f} mF1 | "
                f"{r['OA']:.4f} OA | "
                f"{r['checkpoint']}\n"
            )

    print(f"✓ Wrote summary: {out_path.resolve()}")


if __name__ == "__main__":
    main()
