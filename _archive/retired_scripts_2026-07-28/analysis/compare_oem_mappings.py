#!/usr/bin/env python3
"""
Compare the grounded OEM -> Biodiversity mappings derived from different training sets.

The pre-train mapping is the argmax of each row of the teacher-vs-GT confusion
(scripts/analysis/teacher_oem_to_gt_confusion.py), i.e. "of the pixels the teacher calls OEM class
X, which Biodiversity class are they most often really?". That argmax is what
`geoseg.taxonomy.OEM_TO_STUDENT_PRETRAIN` hard-codes, and it determines how the OEM tiles are
relabelled for pre-training.

The campaign grounds the mapping ONCE per experiment on the intersection of its splits' training
sets. This script checks the assumption that makes that safe and cheap: that the mapping does not
actually depend on which split you measure it over. Feed it the intersection-grounded matrix plus
the per-split matrices.

  - mappings all identical  -> the harmonisation is not split-dependent. The existing relabelled OEM
    data and pre-training pools stay valid, and that invariance is worth stating in the manuscript.
  - any mapping differs     -> the OEM relabel (A8) and every combined pool (A10) must be rebuilt
    per split, and the mapping becomes a factor that varies across the factorial. Exits non-zero.

Also checks each matrix against the hard-coded OEM_TO_STUDENT_PRETRAIN, since RUNBOOK A0 asserts
they agree and a silent divergence there would mis-relabel the pre-training data.

Run:
    PYTHONPATH=. python scripts/analysis/compare_oem_mappings.py \
        artifacts/teacher_oem_gt_confusion_primary.npz \
        artifacts/teacher_oem_gt_confusion_a1.npz ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from geoseg.taxonomy import (  # noqa: E402
    OEM_NATIVE_CLASSES, STUDENT_CLASSES, OEM_TO_STUDENT_PRETRAIN,
)


def mapping_from(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Row-argmax mapping and the row-normalised confidence behind each choice."""
    d = np.load(npz_path, allow_pickle=True)
    hard = d["hard"].astype(np.float64)               # (9 OEM, 6 student)
    totals = hard.sum(axis=1, keepdims=True)
    conf = np.divide(hard, np.where(totals > 0, totals, 1))
    mapping = hard.argmax(axis=1)
    mapping[totals[:, 0] == 0] = -1                   # OEM class the teacher never predicted
    return mapping, conf[np.arange(len(mapping)), np.clip(mapping, 0, None)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("matrices", nargs="+", help="teacher_oem_gt_confusion*.npz to compare")
    ap.add_argument("--margin-warn", type=float, default=0.05,
                    help="warn when the argmax beats the runner-up by less than this")
    args = ap.parse_args()

    maps, confs, names = {}, {}, []
    for p in args.matrices:
        path = Path(p)
        if not path.exists():
            print(f"MISSING: {p}")
            return 2
        name = path.stem.replace("teacher_oem_gt_confusion", "").strip("_") or "default"
        names.append(name)
        maps[name], confs[name] = mapping_from(path)

    width = max(len(n) for n in names) + 2
    print(f"{'OEM class':<18}" + "".join(f"{n:<{width}}" for n in names) + "  hard-coded")
    print("-" * (18 + width * len(names) + 12))
    for o, oem in enumerate(OEM_NATIVE_CLASSES):
        cells = []
        for n in names:
            m = maps[n][o]
            cells.append("(never pred)" if m < 0 else f"{STUDENT_CLASSES[m]}")
        hc = STUDENT_CLASSES[OEM_TO_STUDENT_PRETRAIN[o]]
        flag = "" if len(set(cells)) == 1 else "   <-- DIFFERS"
        print(f"{oem:<18}" + "".join(f"{c:<{width}}" for c in cells) + f"  {hc}{flag}")

    ref = names[0]
    differing = [n for n in names[1:] if not np.array_equal(maps[n], maps[ref])]
    # OEM_TO_STUDENT_PRETRAIN is a dict keyed by OEM index, not a sequence -- np.asarray on it
    # yields a 0-d object array, so index it explicitly.
    hardcoded = np.array([OEM_TO_STUDENT_PRETRAIN[o] for o in range(len(OEM_NATIVE_CLASSES))])
    vs_hardcoded = [n for n in names
                    if not np.array_equal(np.where(maps[n] < 0, hardcoded, maps[n]), hardcoded)]

    print()
    low = [(n, OEM_NATIVE_CLASSES[o], confs[n][o])
           for n in names for o in range(len(OEM_NATIVE_CLASSES))
           if 0 <= maps[n][o] and confs[n][o] < 0.5]
    if low:
        print("Low-confidence argmax rows (winner below 50% of the row):")
        for n, oem, c in low:
            print(f"  {n}: {oem} at {100*c:.1f}%")

    if differing:
        print(f"\nFAIL: mapping differs between grounding sets: {differing} vs '{ref}'.")
        print("  -> the OEM relabel (A8) and every combined pool (A10) must be rebuilt per split,")
        print("     and the mapping varies across the factorial. Do not launch the campaign.")
        return 1
    if vs_hardcoded:
        print(f"\nFAIL: {vs_hardcoded} disagree with geoseg.taxonomy.OEM_TO_STUDENT_PRETRAIN.")
        print("  -> taxonomy.py must be updated and the OEM data relabelled before training.")
        return 1

    print("PASS: every grounding set yields the same mapping, and it matches "
          "geoseg.taxonomy.OEM_TO_STUDENT_PRETRAIN.")
    print("  -> the harmonisation is not split-dependent; the relabelled OEM data and the "
          "pre-training pools stay valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
