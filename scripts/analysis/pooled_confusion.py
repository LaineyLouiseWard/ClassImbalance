#!/usr/bin/env python3
"""
Pooled foreground confusion, per cell, over the seeds and the deduplicated scoring subset.

`compute_metrics.py` writes a confusion per checkpoint. This writes ONE matrix per cell, summed over
seeds, on the same non-overlapping tile subset the reported IoU uses, so the class-pair statistics in
`narrative_numbers.py` are a census of the same ground as the accuracy table. Producing it any other
way is what left the 47% pair share and the reported IoU on two different tile populations.

Rows are the reference class, columns the prediction, both on the six-class index. Only tiles in
`artifacts/scoring_subset_<tag>.json` are counted, and only pixels whose reference is foreground.

Run:
    PYTHONPATH=. python scripts/analysis/pooled_confusion.py \\
        --softmax-root <dir> --split test --cell stage1_baseline --out-dir analysis/confusion
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

SPLIT_TAG = os.environ.get("SPLIT_TAG", "f1")
C = 6
SEEDS = list(range(42, 52))


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        # `artifacts` and `scripts`, NOT `data`: data/ is gitignored, so keying on it made every
        # one of these scripts raise "repo root not found" in a fresh clone -- including the ledger
        # a reviewer would run to check the paper's numbers.
        if (parent / "artifacts").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--softmax-root", required=True)
    ap.add_argument("--split", default="test", choices=["test", "external_test"])
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--all-tiles", action="store_true",
                    help="score every tile instead of the deduplicated subset. Off by default: the "
                         "reported IoU uses the subset, and mixing the two silently compares "
                         "different populations of ground.")
    ap.add_argument("--out-dir", default="analysis/confusion")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(find_repo_root()))
    from scripts.analysis.seed_disagreement import load_seed_stack   # noqa: E402

    root = find_repo_root()
    mask_dir = root / f"data/split_{SPLIT_TAG}/{args.split}/masks"
    keep = None
    if not args.all_tiles:
        sub = json.loads((root / f"artifacts/scoring_subset_{SPLIT_TAG}.json").read_text())
        keep = set(sub["splits"][args.split]["tiles"])

    tiles = sorted(p for p in mask_dir.glob("*.png") if keep is None or p.stem in keep)
    if keep is not None and len(tiles) != len(keep):
        raise SystemExit(f"expected {len(keep)} subset tiles in {mask_dir}, found {len(tiles)}. A "
                         f"partially staged mask directory would silently score less ground.")
    if not tiles:
        raise SystemExit(f"no tiles to score in {mask_dir} under the "
                         f"{'full split' if keep is None else 'scoring subset'}")

    conf = np.zeros((C, C), dtype=np.int64)
    per_seed = np.zeros((len(args.seeds), C, C), dtype=np.int64)
    for p in tiles:
        m = np.array(Image.open(p).convert("L"))
        fg = m != 0
        if not fg.any():
            continue
        stack = load_seed_stack(args.softmax_root, args.seeds, args.cell, p.stem)
        for si in range(len(args.seeds)):
            pred = stack[si].argmax(axis=0)
            np.add.at(per_seed[si], (m[fg].ravel(), pred[fg].ravel()), 1)
    conf = per_seed.sum(axis=0)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"confusion_{args.split}_{args.cell}.npy", conf)
    np.save(out / f"confusion_per_seed_{args.split}_{args.cell}.npy", per_seed)
    meta = {"split": args.split, "cell": args.cell, "seeds": args.seeds,
            "n_tiles": len(tiles), "scoring_subset": not args.all_tiles,
            "foreground_px": int(conf.sum()),
            # Foreground reference pixels whose prediction differs. Row 0 is identically zero
            # because only `m != 0` pixels are counted, so conf.sum() is already foreground-only;
            # the earlier form added and subtracted conf[1:,0].sum() and reduced to this.
            "foreground_errors": int(conf[1:, :].sum() - np.trace(conf[1:, 1:]))}
    (out / f"confusion_{args.split}_{args.cell}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"wrote {out}/confusion_{args.split}_{args.cell}.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
