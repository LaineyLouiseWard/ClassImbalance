# If there is time

Improvements that are worth having and are not worth risking a deadline for. Nothing here blocks the
paper. Nothing here is a known defect — each one makes an existing result stronger or removes a question
a reviewer might raise.

Ordered by value for effort.

**Do 1 before 2, and do not do 2 without 1.** They fix different things — width is about a patch's shape,
mosaicking is about how pixels are grouped into patches — and mosaicking on its own makes the area
evidence weaker rather than stronger. If any of the error runs along class boundaries, and boundaries
form one connected network across the map, then merging across tile edges joins those ribbon segments
into a single enormous component. Its area climbs steeply while it stays a ribbon. The tiling is
currently capping that by accident. Width is also the more robust measurement of the two: for a genuinely
wide patch the largest inscribed circle usually sits well inside it, so clipping a corner at a tile edge
often leaves the width alone, where it changes the area directly.

## 1. Measure patch width, not only patch area

**What.** `component_sizes` returns pixel count × pixel area and nothing else, so a long thin ribbon and
a compact blob of equal area score the same. Width comes from a Euclidean distance transform run inside
each patch: for every pixel, the distance to the nearest pixel outside the patch. The largest such
distance is the radius of the biggest circle that fits, so twice it is the width at the widest point. A
two-metre ribbon scores about a metre however far it runs; a one-hectare blob scores tens of metres.

**Why it matters.** It separates ribbon from blob directly instead of by the hectare threshold, which is
a proxy. The threshold was moved from 0.1 ha to 1 ha precisely because the smaller one could not tell
the two apart.

**Watch for.** Patches touching a tile edge have no background beyond it, so their width may come out
inflated. Check how the distance transform behaves at array edges before trusting the numbers.

**Honest caveat.** This refines a supporting result, not the lead. The paper's main evidence — that the
two classes barely share a border — needs no patch analysis at all.

## 2. Mosaic the predictions before measuring error patches

**What.** Connected components of the error mask are currently labelled per 512×512 tile, so every patch
is cut at a tile edge and no patch can exceed 6.55 ha. Reassemble the predictions onto real map
coordinates first, then label components.

**Why it is safe to leave.** Merging can only make patches bigger — the set of wrong pixels does not
change, only how it is grouped. So every size reported now is a lower bound, and the share of error mass
in patches over a hectare can only rise if this is done. Say that in Methods and the limitation is
answered by an argument rather than a hedge.

**Why it is still worth doing.** The number gets larger, and one obvious referee question disappears.

**Cost and risk.** The per-seed predictions are on Sonic (`scratch/lqc/dedup_root`), fast to fetch from
campus. The risk is not the fetch — it is that the share may move a long way once patches merge across
tiles, and then every ledger row and every sentence quoting it moves with it. Do this when a draft
exists, never mid-integration.

**One thing it does not buy.** A merged patch is still not a field. Claiming "field" needs the reference
parcel boundaries, which is a different measurement.

**First check before starting:** are the 90 scored chips contiguous on the ground, or are there gaps
between them? They were selected to be non-overlapping, and if that left gaps, mosaicking still cuts
patches at every gap edge and gains much less. Read the bounds from
`data/biodiversity_raw/images/*.tif` for the tiles in `artifacts/scoring_subset_f1.json`.

## 3. Close the ledger rows

Quoted in the narrative, no row in `verify_narrative_numbers.py` yet: the class adjacency figures, the
per-direction rate ratios, the 15.0%, and the 72.4% background. A number with no row is a number nobody
can defend.

## 4. Check the adjacency measure for a tile-edge guard

The component statistic excludes pixels near a tile edge, where something just outside the tile cannot be
seen. It is not established that the adjacency measure does the same. If it does not, semi-natural
sitting just beyond a tile edge goes uncounted and "under 1% of grassland within eight metres of
semi-natural" would be understated.

Understated is the harmless direction for that claim — it makes the bound conservative — but this is the
paper's lead evidence, so it should be confirmed rather than assumed. Noted 2026-07-29, unverified.
