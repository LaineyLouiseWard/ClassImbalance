#!/usr/bin/env python3
"""
Do the classes the model confuses actually touch each other on the ground?

The boundary analyses ask how far an error is from the NEAREST BOUNDARY OF ANY KIND. That cannot
distinguish "semi-natural is misread along its edge with grassland" from "whole blocks of
semi-natural are called grassland nowhere near any grassland". This script asks the second question
directly, from the reference masks alone -- no predictions enter the geometry.

Three quantities, per ordered class pair (A, B):

  contacts(A,B)   four-neighbour adjacencies between A and B pixels, as a share of all
                  foreground-to-foreground adjacencies. How much the two classes touch at all.
  near(A|B)       share of A's pixels lying within 8 m of ANY B pixel, using the per-site
                  anisotropic sampling. The opportunity for an edge-driven A->B error.
  bound(A->B)     given the model calls a fraction r of A's pixels B, AT MOST near(A|B)/r of those
                  errors can lie within 8 m of a B pixel. The rest are not near the other class.

`bound` is the point. It is an upper bound on how much of a confusion can be edge-driven, derived
from geometry and one confusion rate, and it needs no assumption about where the errors sit.

Run:
    PYTHONPATH=. python scripts/analysis/class_adjacency.py --self-test
    PYTHONPATH=. python scripts/analysis/class_adjacency.py \\
        --split test --confusion analysis/confusion/confusion_test_stage1_baseline.npy
"""
from __future__ import annotations

import argparse
import json
import os
from itertools import permutations
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

SPLIT_TAG = os.environ.get("SPLIT_TAG", "f1")
FG = {1: "Forest", 2: "Grassland", 3: "Cropland", 4: "Settlement", 5: "Seminatural"}
NEAR_M = 8.0


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "data").is_dir() and (parent / "artifacts").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def contacts(mask: np.ndarray, acc: np.ndarray) -> None:
    """Count four-neighbour adjacencies between differing foreground classes into acc[a, b].

    Each adjacent unordered pair is counted once in each direction, so acc is symmetric.
    """
    for a, b in ((mask[:, :-1], mask[:, 1:]), (mask[:-1, :], mask[1:, :])):
        sel = (a != b) & (a != 0) & (b != 0)
        if sel.any():
            np.add.at(acc, (a[sel].ravel(), b[sel].ravel()), 1)
            np.add.at(acc, (b[sel].ravel(), a[sel].ravel()), 1)


def physical_contacts(acc: np.ndarray) -> int:
    """Number of distinct foreground adjacencies in a symmetric contact matrix.

    `contacts` writes every adjacency into BOTH acc[a,b] and acc[b,a], so the matrix sum is twice
    the number of physical contacts while each per-pair numerator counts it once. Dividing one by
    the other halved every reported share until 2026-07-28. This lives in a function so the
    self-test exercises the same code main() does -- an earlier gate recomputed the halving inline
    and therefore passed when main()'s copy was reverted.
    """
    return int(acc[1:, 1:].sum()) // 2


def near_counts(mask: np.ndarray, sampling, near_n: np.ndarray, tot_n: np.ndarray) -> None:
    """near_n[a, b] += pixels of class a within NEAR_M of any pixel of class b."""
    present = [k for k in FG if (mask == k).any()]
    for k in present:
        tot_n[k] += int((mask == k).sum())
    dist = {}
    for b in present:
        # EDT of the complement gives, per pixel, distance to the nearest True in (mask == b).
        dist[b] = ndimage.distance_transform_edt(mask != b, sampling=sampling)
    for a in present:
        sel = mask == a
        for b in present:
            if a == b:
                continue
            near_n[a, b] += int((sel & (dist[b] < NEAR_M)).sum())


def self_test() -> int:
    """Two slabs that touch, and an island that does not, with the answers known by construction."""
    ok = True
    m = np.zeros((200, 200), np.uint8)
    m[:, :100] = 1                     # forest slab, left half
    m[:, 100:] = 2                     # grassland slab, right half
    m[20:40, 20:40] = 5                # semi-natural island, deep inside forest, far from grassland

    acc = np.zeros((6, 6), np.int64)
    contacts(m, acc)
    # The 1|2 seam is one column boundary over 200 rows: 200 adjacent pairs, written into both
    # acc[1,2] and acc[2,1]. (The first version of this gate expected 400 in each cell, which
    # double-counted the symmetry; the gate failed and the expectation was wrong, not the code.)
    good = acc[1, 2] == 200 and acc[1, 2] == acc[2, 1]
    ok &= good
    print(f"  forest-grassland contacts {acc[1,2]} (expect 200, symmetric)  [{'ok' if good else 'FAIL'}]")
    # The island touches only forest, never grassland.
    # EXACT count, not `> 0`. The 20x20 island has 40 horizontal and 40 vertical adjacencies with
    # the surrounding forest, so dropping either axis of `contacts` halves it. A bare `> 0` survived
    # exactly that mutation, and the fixture's only class seam is vertical, so nothing else in this
    # test looked at the horizontal pass.
    good = acc[5, 2] == 0 and acc[5, 1] == 80
    ok &= good
    print(f"  island touches grassland {acc[5,2]} (expect 0) and forest {acc[5,1]} (expect 80: "
          f"40 horizontal + 40 vertical)  [{'ok' if good else 'FAIL'}]")
    # The denominator counts PHYSICAL contacts, so it is half the symmetric matrix sum. Without
    # this, reverting that division leaves every reported contact share exactly halved and the
    # self-test silent.
    total = physical_contacts(acc)
    good = total == 280
    ok &= good
    print(f"  physical contacts {total} (expect 280 = 200 seam + 80 island)  "
          f"[{'ok' if good else 'FAIL'}]")
    shares = sum(acc[a, b] for a in FG for b in FG if a < b) / total
    good = abs(shares - 1.0) < 1e-9
    ok &= good
    print(f"  unordered shares sum to {shares:.6f} (expect 1.000000)  [{'ok' if good else 'FAIL'}]")

    near_n = np.zeros((6, 6), np.int64)
    tot_n = np.zeros(6, np.int64)
    near_counts(m, (0.5, 0.5), near_n, tot_n)
    # The island spans x in [20,40); grassland starts at x=100. Nearest grassland is 60 px = 30 m.
    good = near_n[5, 2] == 0
    ok &= good
    print(f"  island pixels within 8 m of grassland: {near_n[5,2]} (expect 0, it is 30 m away)  "
          f"[{'ok' if good else 'FAIL'}]")
    # Forest within 8 m of grassland at 0.5 m/px: grassland starts at column 100, so column c is
    # (100-c)*0.5 m away and STRICT `< 8.0` admits 100-c < 16, i.e. columns 85..99 -- fifteen
    # columns, not sixteen. Getting this off by one is exactly the strict-membership error the
    # boundary code has its own gate for.
    expect = 15 * 200
    good = near_n[1, 2] == expect
    ok &= good
    print(f"  forest within 8 m of grassland: {near_n[1,2]} (expect {expect})  "
          f"[{'ok' if good else 'FAIL'}]")

    # And the bound must be vacuous when the classes are adjacent, tight when they are not.
    b_far = 1.0 - (near_n[5, 2] / max(tot_n[5], 1)) / 0.40
    good = abs(b_far - 1.0) < 1e-9
    ok &= good
    print(f"  with 40% of the island called grassland, >= {100*b_far:.1f}% of those errors are far "
          f"(expect 100.0)  [{'ok' if good else 'FAIL'}]")

    print("\nSELF-TEST PASSED" if ok else "\nSELF-TEST FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "external_test"])
    ap.add_argument("--dedup", action="store_true", default=True)
    ap.add_argument("--all-tiles", dest="dedup", action="store_false")
    ap.add_argument("--confusion", help="6x6 .npy from pooled_confusion.py, for the bound")
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    root = find_repo_root()
    import sys
    sys.path.insert(0, str(root))
    if args.self_test:
        return self_test()

    from geoseg.geo import gsd_for

    mask_dir = root / f"data/split_{SPLIT_TAG}/{args.split}/masks"
    keep = None
    if args.dedup:
        sub = json.loads((root / f"artifacts/scoring_subset_{SPLIT_TAG}.json").read_text())
        keep = set(sub["splits"][args.split]["tiles"])
    tiles = sorted(p for p in mask_dir.glob("*.png") if keep is None or p.stem in keep)

    acc = np.zeros((6, 6), np.int64)
    near_n = np.zeros((6, 6), np.int64)
    tot_n = np.zeros(6, np.int64)
    for p in tiles:
        m = np.array(Image.open(p).convert("L"))
        if not (m != 0).any():
            continue
        contacts(m, acc)
        near_counts(m, gsd_for(p.stem), near_n, tot_n)

    # acc is symmetric -- every physical adjacency is written into BOTH acc[a,b] and acc[b,a] -- so
    # summing the matrix counts each contact twice. The per-pair numerator acc[a,b] counts it once.
    # Dividing one by the other halved every contact share until 2026-07-28; an independent
    # re-derivation from the masks gave 1.59% for the grassland pair where this reported 0.80%.
    total_contacts = physical_contacts(acc)
    out = {"split": args.split, "dedup": bool(args.dedup), "n_tiles": len(tiles),
           "near_m": NEAR_M, "total_fg_contacts": total_contacts, "pairs": []}

    conf = np.load(args.confusion) if args.confusion else None
    for a, b in permutations(FG, 2):
        if a > b and conf is None:
            continue
        share_contacts = acc[a, b] / total_contacts if total_contacts else float("nan")
        near_frac = near_n[a, b] / max(tot_n[a], 1)
        rec = {"a": FG[a], "b": FG[b], "contacts": int(acc[a, b]),
               "contact_share": share_contacts,
               "share_of_a_within_8m_of_b": near_frac,
               "a_pixels": int(tot_n[a])}
        if conf is not None:
            # Contact-based null: what share of foreground error would this pair carry if error fell
            # in proportion to how much the two classes actually TOUCH? The co-area null in
            # narrative_numbers.py asks the same question of area instead. Reported here so the
            # 29x figure has a script behind it; the co-area figure stays the one quoted, being
            # roughly fourteenfold more conservative.
            tot_err = float(conf[1:, :].sum() - np.trace(conf[1:, 1:]))
            pair_err = float(conf[a, b] + conf[b, a])
            rec["pair_share_of_error"] = pair_err / tot_err if tot_err else float("nan")
            rec["expected_share_from_contacts"] = (acc[a, b] / total_contacts) if total_contacts else float("nan")
            rec["contact_null_ratio"] = (rec["pair_share_of_error"] / rec["expected_share_from_contacts"]
                                         if rec["expected_share_from_contacts"] else float("nan"))
            r = conf[a, b] / max(conf[a, :].sum(), 1)
            rec["rate_a_called_b"] = float(r)
            rec["max_share_of_those_errors_near_b"] = float(min(1.0, near_frac / r)) if r > 0 else None
            rec["min_share_far_from_b"] = (1.0 - min(1.0, near_frac / r)) if r > 0 else None
        out["pairs"].append(rec)

    p = Path(args.out) if args.out else root / f"artifacts/class_adjacency_{args.split}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))

    print(f"{args.split}: {len(tiles)} tiles, {total_contacts:,} foreground-foreground contacts")
    rows = sorted(out["pairs"], key=lambda d: -(d.get("rate_a_called_b") or 0))
    print(f"{'pair':26s}{'contact%':>10s}{'A within 8m of B':>18s}{'A called B':>12s}{'>= far':>9s}")
    for r in rows[:8]:
        far = r.get("min_share_far_from_b")
        print(f"{r['a']}->{r['b']:<18s}{100*r['contact_share']:9.2f}%"
              f"{100*r['share_of_a_within_8m_of_b']:17.2f}%"
              f"{100*(r.get('rate_a_called_b') or 0):11.2f}%"
              f"{('%.1f%%' % (100*far)) if far is not None else '-':>9s}")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
