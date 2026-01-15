#!/usr/bin/env python3
"""
evaluation/tile_number_metrics.py

Counts dataset tile totals for the *current ClassImbalance data tree*:

data/
  biodiversity_raw/{images,masks}
  biodiversity_split/{train,train_rep,val,test}/{images,masks}
  biodiversity_oem_combined/{train,val,test}/{images,masks}
  openearthmap_raw/OpenEarthMap/...            (raw download; recursive count)
  openearthmap_filtered/{images,masks}         (rural-filtered OEM)
  openearthmap_relabelled/{images,masks}       (OEM mapped to Biodiversity taxonomy)
  openearthmap_teacher/{train,val}/{images,masks} (teacher dataset split)

Outputs:
- A human-readable report to stdout.
- Optional JSON report (for copy/paste into paper later).

Notes:
- “paired tiles” = image+mask share same stem.
- “unique IDs” strips replication suffix *_repN.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


REP_SUFFIX_RE = re.compile(r"_rep\d+$", re.IGNORECASE)


# -------------------------
# helpers
# -------------------------
def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "data").exists() and (p / "evaluation").exists():
            return p
    return start


def list_files(dir_path: Path, exts: Sequence[str], recursive: bool = False) -> List[Path]:
    if not dir_path.exists():
        return []
    files: List[Path] = []
    if recursive:
        for ext in exts:
            files.extend(dir_path.rglob(f"*{ext}"))
    else:
        for ext in exts:
            files.extend(dir_path.glob(f"*{ext}"))
    return sorted(files)


def stems(paths: Iterable[Path]) -> Set[str]:
    return {p.stem for p in paths}


def stem_no_rep(stem: str) -> str:
    return REP_SUFFIX_RE.sub("", stem)


def fmt(n: int) -> str:
    return f"{n:,}"


@dataclass
class DirCounts:
    name: str
    root: str
    images: int
    masks: int
    paired: int
    unique_ids: int
    replicas: int


def count_paired_dir(
    name: str,
    root: Path,
    img_dir: str = "images",
    mask_dir: str = "masks",
    img_exts: Sequence[str] = (".tif", ".tiff", ".png", ".jpg", ".jpeg"),
    mask_exts: Sequence[str] = (".png",),
    recursive_images: bool = False,
    recursive_masks: bool = False,
) -> DirCounts:
    img_p = root / img_dir
    msk_p = root / mask_dir

    imgs = list_files(img_p, img_exts, recursive=recursive_images)
    msks = list_files(msk_p, mask_exts, recursive=recursive_masks)

    img_ids = stems(imgs)
    msk_ids = stems(msks)
    paired_ids = img_ids & msk_ids

    uniq = {stem_no_rep(s) for s in paired_ids}
    replicas = len(paired_ids) - len(uniq)

    return DirCounts(
        name=name,
        root=str(root),
        images=len(imgs),
        masks=len(msks),
        paired=len(paired_ids),
        unique_ids=len(uniq),
        replicas=replicas,
    )


def print_block(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def print_counts_row(c: DirCounts) -> None:
    print(
        f"{c.name:<32} paired={fmt(c.paired):>6}  "
        f"(images={fmt(c.images):>6}, masks={fmt(c.masks):>6}, "
        f"unique={fmt(c.unique_ids):>6}, reps={fmt(c.replicas):>5})"
    )


# -------------------------
# main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data", help="Relative to repo root")
    ap.add_argument("--out-json", default="", help="Optional JSON output path, e.g. artifacts/tile_number_metrics.json")
    args = ap.parse_args()

    repo_root = find_repo_root(Path.cwd())
    data_root = (repo_root / args.data_root).resolve()

    # ---- A) Raw dataset sizes (provenance) ----
    biodiv_raw = count_paired_dir(
        name="Biodiversity raw",
        root=data_root / "biodiversity_raw",
        img_dir="images",
        mask_dir="masks",
        img_exts=(".tif", ".tiff", ".png", ".jpg", ".jpeg"),
        mask_exts=(".png",),
    )

    # OEM raw is nested; count recursively under openearthmap_raw/OpenEarthMap
    oem_raw_root = data_root / "openearthmap_raw" / "OpenEarthMap"
    oem_raw = count_paired_dir(
        name="OpenEarthMap raw",
        root=oem_raw_root,
        img_dir="images",
        mask_dir="masks",
        img_exts=(".tif", ".tiff", ".png", ".jpg", ".jpeg"),
        mask_exts=(".png",),
        recursive_images=True,
        recursive_masks=True,
    )

    # ---- B) OEM processing ----
    oem_filtered = count_paired_dir(
        name="OEM rural-filtered",
        root=data_root / "openearthmap_filtered",
    )
    oem_relabelled = count_paired_dir(
        name="OEM relabelled (→Biodiv taxonomy)",
        root=data_root / "openearthmap_relabelled",
    )

    # ---- C) OEM teacher dataset ----
    oem_teacher_train = count_paired_dir(
        name="OEM teacher train",
        root=data_root / "openearthmap_teacher" / "train",
    )
    oem_teacher_val = count_paired_dir(
        name="OEM teacher val",
        root=data_root / "openearthmap_teacher" / "val",
    )

    # ---- D) Biodiversity split + replication ----
    biodiv_split_root = data_root / "biodiversity_split"
    biodiv_train = count_paired_dir(name="Biodiversity train", root=biodiv_split_root / "train")
    biodiv_train_rep = count_paired_dir(name="Biodiversity train_rep", root=biodiv_split_root / "train_rep")
    biodiv_val = count_paired_dir(name="Biodiversity val", root=biodiv_split_root / "val")
    biodiv_test = count_paired_dir(name="Biodiversity test", root=biodiv_split_root / "test")

    # ---- E) Combined pretraining pool (Biodiversity + OEM-filtered) ----
    combined_root = data_root / "biodiversity_oem_combined"
    combined_train = count_paired_dir(name="Combined pretrain train", root=combined_root / "train")
    combined_val = count_paired_dir(name="Combined pretrain val", root=combined_root / "val")
    combined_test = count_paired_dir(name="Combined pretrain test", root=combined_root / "test")

    # ---- Print report ----
    print(f"Repo root: {repo_root}")
    print(f"Data root: {data_root}")

    print_block("A) Raw dataset sizes (provenance)")
    print_counts_row(biodiv_raw)
    print_counts_row(oem_raw)

    print_block("B) OpenEarthMap processing (student-side)")
    print_counts_row(oem_filtered)
    print_counts_row(oem_relabelled)

    print_block("C) OpenEarthMap teacher dataset")
    print_counts_row(oem_teacher_train)
    print_counts_row(oem_teacher_val)

    print_block("D) Biodiversity splits (training / eval)")
    print_counts_row(biodiv_train)
    print_counts_row(biodiv_train_rep)
    print_counts_row(biodiv_val)
    print_counts_row(biodiv_test)

    print_block("E) Combined pretraining pool (Biodiversity + OEM)")
    print_counts_row(combined_train)
    print_counts_row(combined_val)
    print_counts_row(combined_test)

    # ---- Stage mapping (paper-facing) ----
    print_block("F) Stage-wise pools (paper-facing)")
    # Stages 1–4: Biodiversity (train or train_rep), Stage 5: combined pretrain then finetune, Stage 6: distill finetune
    print(f"{'Stage 1–2':<12} Biodiversity train/train_rep pool: {fmt(biodiv_train.paired)} / {fmt(biodiv_train_rep.paired)} paired tiles")
    print(f"{'Stage 3–4':<12} Biodiversity train_rep pool:       {fmt(biodiv_train_rep.paired)} paired tiles (same pool; different sampling/cropping)")
    print(f"{'Stage 5 pre':<12} Combined pretrain train pool:     {fmt(combined_train.paired)} paired tiles")
    print(f"{'Stage 5 ft':<12} Biodiversity train_rep pool:       {fmt(biodiv_train_rep.paired)} paired tiles")
    print(f"{'Stage 6':<12} Biodiversity train_rep pool:       {fmt(biodiv_train_rep.paired)} paired tiles (teacher supervision)")
    print(f"{'Eval':<12} Biodiversity val/test pools:       {fmt(biodiv_val.paired)} / {fmt(biodiv_test.paired)} paired tiles")

    # ---- Optional JSON ----
    if args.out_json:
        out_path = (repo_root / args.out_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "repo_root": str(repo_root),
            "data_root": str(data_root),
            "counts": {
                "biodiversity_raw": asdict(biodiv_raw),
                "openearthmap_raw": asdict(oem_raw),
                "openearthmap_filtered": asdict(oem_filtered),
                "openearthmap_relabelled": asdict(oem_relabelled),
                "openearthmap_teacher_train": asdict(oem_teacher_train),
                "openearthmap_teacher_val": asdict(oem_teacher_val),
                "biodiversity_train": asdict(biodiv_train),
                "biodiversity_train_rep": asdict(biodiv_train_rep),
                "biodiversity_val": asdict(biodiv_val),
                "biodiversity_test": asdict(biodiv_test),
                "combined_train": asdict(combined_train),
                "combined_val": asdict(combined_val),
                "combined_test": asdict(combined_test),
            },
            "stage_pools": {
                "stage_1_train": biodiv_train.paired,
                "stage_2_train_rep": biodiv_train_rep.paired,
                "stage_3_train_rep": biodiv_train_rep.paired,
                "stage_4_train_rep": biodiv_train_rep.paired,
                "stage_5_pretrain_combined_train": combined_train.paired,
                "stage_5_finetune_train_rep": biodiv_train_rep.paired,
                "stage_6_distill_train_rep": biodiv_train_rep.paired,
                "eval_val": biodiv_val.paired,
                "eval_test": biodiv_test.paired,
            },
        }

        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {out_path}")


if __name__ == "__main__":
    main()
