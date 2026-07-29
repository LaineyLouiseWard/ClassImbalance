#!/usr/bin/env python3
"""Per-chip grassland / semi-natural error statistics, for one cell and one seed.

The pooled confusion says how much error the pair carries. This says how it is DISTRIBUTED over
the ninety Test A scoring chips, which is what the qualitative figure's chip selection and the
"chips with no semi-natural at all" figure both rest on.

Reference-foreground pixels only, matching pooled_confusion.py. Two distances are easy to confuse
and this script computes only the second: `pair_error_geometry.py` measures each error pixel's
distance to the nearest reference boundary OF ANY KIND, with a tile-edge guard; the `far8` fields
here measure distance to the nearest pixel OF THE CLASS IT WAS CONFUSED WITH, unguarded. They
answer different questions and their values differ by tens of points -- do not quote one for the
other.

Run on the cluster where the softmax dumps live:
    python scripts/analysis/pair_error_by_tile.py <masks-dir> <softmax-dir> <out.json>

Distances use the inland 0.5 m isotropic sampling; the script refuses any chip id that is not from
the inland site, so an anisotropic upland chip cannot silently rescale every band.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

MASKS = Path(sys.argv[1])
SOFTMAX = Path(sys.argv[2])
OUT = Path(sys.argv[3])
G, S = 2, 5
NEAR_M = 8.0
PX = 0.5

rows = []
for p in sorted(MASKS.glob("*.png")):
    tid = p.stem
    if not tid.startswith("biodiversity_"):
        raise SystemExit(f"{tid}: not the inland site; 0.5 m sampling would be wrong")
    m = np.array(Image.open(p).convert("L"))
    fg = m != 0
    if not fg.any():
        continue
    sm = np.load(SOFTMAX / f"{tid}.npy")
    pred = sm.argmax(0).astype(np.uint8)

    err = fg & (pred != m)
    g2s = (m == G) & (pred == S)
    s2g = (m == S) & (pred == G)
    pair = g2s | s2g

    rec = {
        "tile": tid,
        "fg_px": int(fg.sum()),
        "g_px": int((m == G).sum()),
        "s_px": int((m == S).sum()),
        "fg_err_px": int(err.sum()),
        "g2s_px": int(g2s.sum()),
        "s2g_px": int(s2g.sum()),
    }

    # Distance from each pair-error pixel to the nearest pixel of the class it was confused with.
    if g2s.any():
        d = ndimage.distance_transform_edt(m != S, sampling=(PX, PX))
        rec["g2s_far8_px"] = int((g2s & (d >= NEAR_M)).sum())
    else:
        rec["g2s_far8_px"] = 0
    if s2g.any():
        d = ndimage.distance_transform_edt(m != G, sampling=(PX, PX))
        rec["s2g_far8_px"] = int((s2g & (d >= NEAR_M)).sum())
    else:
        rec["s2g_far8_px"] = 0

    # Largest connected block of pair error, in hectares.
    if pair.any():
        lab, n = ndimage.label(pair)
        sizes = np.bincount(lab.ravel())[1:]
        rec["pair_components"] = int(n)
        rec["pair_largest_ha"] = float(sizes.max() * PX * PX / 10000.0)
    else:
        rec["pair_components"] = 0
        rec["pair_largest_ha"] = 0.0
    rows.append(rec)

OUT.write_text(json.dumps({"masks": str(MASKS), "softmax": str(SOFTMAX),
                           "near_m": NEAR_M, "tiles": rows}, indent=2))
print(f"wrote {OUT}: {len(rows)} tiles")
