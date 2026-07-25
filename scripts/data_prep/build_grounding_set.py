#!/usr/bin/env python3
"""
Build the tile set used to ground the OpenEarthMap -> Biodiversity taxonomy mapping.

The mapping is the argmax of the frozen OEM teacher's confusion against Biodiversity ground truth
(scripts/analysis/teacher_oem_to_gt_confusion.py). Its argmax IS
`geoseg.taxonomy.OEM_TO_STUDENT_PRETRAIN`, which relabels the OEM tiles, which builds the
pre-training pool -- so it feeds the transfer arm's training data and must not see held-out ground.

Grounding it on any single split's training set would make the mapping vary between the very cells
the factorial compares. Instead it is grounded ONCE per experiment, on the INTERSECTION of the
training sets of every split that experiment uses. Those tiles are training data under all of them,
so the mapping is leakage-free for each while staying fixed across them.

  primary experiment (a1,a2,a3) -> intersect 3 training sets  (~801 tiles, includes upland)
  LOSO experiment               -> intersect all 4            (~753 tiles, inland only, which is
                                   also correct: upland is LOSO's test domain)

Asserts the result shares no tile with any contributing split's val/test.

Run:
    PYTHONPATH=. python scripts/data_prep/build_grounding_set.py \
        --manifests artifacts/spatial_split_manifest_a{1,2,3}.json \
        --out-root data/grounding_primary
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "data").is_dir() and (parent / "artifacts").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--split-root", default="data/biodiversity_split",
                    help="source pool the manifests were built from")
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    args = ap.parse_args()

    root = find_repo_root()
    src_root = root / args.split_root

    train_sets, held_sets, orig = [], [], {}
    for m in args.manifests:
        a = json.loads((root / m).read_text())["assignment"]
        train_sets.append({k for k, v in a.items() if v == "train"})
        held_sets.append({k for k, v in a.items() if v in ("val", "test")})
        print(f"  {m}: train={len(train_sets[-1])} held-out={len(held_sets[-1])}")

    inter = set.intersection(*train_sets)
    all_held = set.union(*held_sets)
    bleed = inter & all_held
    if bleed:
        raise SystemExit(f"FAILED: {len(bleed)} grounding tiles are held out in some contributing "
                         f"split, e.g. {sorted(bleed)[:3]}")
    if not inter:
        raise SystemExit("FAILED: the training sets have no tiles in common")

    # locate each tile in the source pool (it may sit under any of train/val/test there)
    for split in ("train", "val", "test"):
        for f in (src_root / split / "images").glob("*.tif"):
            if f.stem in inter:
                orig[f.stem] = split
    missing = inter - set(orig)
    if missing:
        raise SystemExit(f"FAILED: {len(missing)} grounding tiles absent from {src_root}, "
                         f"e.g. {sorted(missing)[:3]}")

    out = root / args.out_root
    if out.exists():
        shutil.rmtree(out)
    for sub in ("images", "masks"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for tid in sorted(inter):
        for sub, ext in (("images", "tif"), ("masks", "png")):
            src = src_root / orig[tid] / sub / f"{tid}.{ext}"
            dst = out / sub / f"{tid}.{ext}"
            if args.mode == "symlink":
                dst.symlink_to(os.path.relpath(src, start=dst.parent))
            else:
                shutil.copy2(src, dst)

    sites = Counter(t.split("_")[0] for t in inter)
    print(f"\ngrounding set: {len(inter)} tiles  by site {dict(sites)}")
    print(f"  shares no tile with any contributing split's val/test  (verified)")
    print(f"wrote {args.out_root} ({args.mode})")

    (root / "artifacts").mkdir(exist_ok=True)
    name = Path(args.out_root).name
    (root / f"artifacts/grounding_set_{name}.json").write_text(json.dumps(
        {"manifests": args.manifests, "n_tiles": len(inter),
         "by_site": dict(sites), "tiles": sorted(inter)}, indent=2))


if __name__ == "__main__":
    main()
