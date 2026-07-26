"""
Canonical taxonomy — the single source of truth for class names, order, colours, and the
OpenEarthMap(OEM)->Biodiversity mappings used across the whole pipeline.

Every other module should DERIVE its class conventions from here rather than re-declaring them,
so the orders/indices can never silently drift (the original KD bug was a stray OEM channel order).
`scripts/verify_taxonomy_consistency.py` asserts that everything still agrees with this module.

This module is pure-Python (no torch / no project imports) so it can be imported anywhere.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Student (Biodiversity) taxonomy — 6 classes, background first.
# ---------------------------------------------------------------------------
STUDENT_CLASSES = (
    "Background",   # 0
    "Forest",       # 1
    "Grassland",    # 2
    "Cropland",     # 3
    "Settlement",   # 4
    "Seminatural",  # 5
)

# RGB palette aligned to STUDENT_CLASSES (index -> colour).
STUDENT_PALETTE = [
    [0, 0, 0],         # 0 Background
    [250, 62, 119],    # 1 Forest
    [168, 232, 84],    # 2 Grassland
    [242, 180, 92],    # 3 Cropland
    [59, 141, 247],    # 4 Settlement
    [255, 214, 33],    # 5 Seminatural
]

BACKGROUND_INDEX = 0
# Minority classes of interest (pixel-rare): Settlement, Semi-natural grassland.
MINORITY_INDICES = (4, 5)

# ---------------------------------------------------------------------------
# Native OpenEarthMap taxonomy — 9 output channels (8 land classes + unknown/background).
# Official OEM integer encoding: 0=unknown, 1=bareland, ... 8=building. The teacher trains
# directly on these labels, so teacher output channel i == OEM class i.
# ---------------------------------------------------------------------------
OEM_NATIVE_CLASSES = (
    "Unknown",       # 0
    "Bareland",      # 1
    "Rangeland",     # 2
    "Developed",     # 3
    "Road",          # 4
    "Tree",          # 5
    "Water",         # 6
    "Agriculture",   # 7
    "Building",      # 8
)

# ---------------------------------------------------------------------------
# OEM native index -> student index for PRE-TRAINING (hard labels). Table 1, pre-training column.
# GROUNDED (2026-06-17): argmax of the teacher's empirical OEM->target confusion on the TRAINING set
# (scripts/analysis/teacher_oem_to_gt_confusion.py). Replaces the original
# name-based hand-map, which mis-routed the teacher's domain-confused classes (Irish "Agriculture" is
# pasture = Grassland, not Cropland; teacher "Water"/"Bareland" land on vegetation, not Background).
# Consistent BY CONSTRUCTION with the KD column (= the full soft confusion): pretrain = argmax(confusion),
# KD = full distribution -- so no "dagger" exceptions are needed anymore.
#   Old name-based hand-map was: {0:0, 1:0, 2:2, 3:4, 4:4, 5:1, 6:0, 7:3, 8:4}
# ---------------------------------------------------------------------------
OEM_TO_STUDENT_PRETRAIN = {
    0: 0,  # Unknown      -> Background    (99% GT Background)
    # REGROUNDED 2026-07-26 on the spatially blocked split: Bareland -> Grassland, was Seminatural.
    # The old grounding (44.2% GT Seminatural, hard) was measured over the leaking split, whose
    # "training" set held 153 of the 191 upland tiles -- and the uplands are ~60% semi-natural. 43.3%
    # of that training set's semi-natural pixel mass came from tiles that are now ENTIRELY in
    # external_test, so the mapping that relabels every OEM pre-training tile was partly grounded on
    # the generalisation test set. Re-measured on the clean training set the same row reads
    # Grassland 61.2%, Cropland 35.6%, Seminatural 0.01% (hard).
    #
    # CONSEQUENCE, and it is a scientific result rather than a config detail: no OEM class now maps to
    # Seminatural, so the transfer arm pre-trains with ZERO semi-natural labels. Any gain it delivers
    # on the priority class is therefore representation transfer, not label transfer.
    1: 2,  # Bareland     -> Grassland     (61% GT Grassland; see the regrounding note above)
    2: 2,  # Rangeland    -> Grassland     (54% GT Grassland)
    3: 4,  # Developed    -> Settlement    (59% GT Settlement)
    4: 4,  # Road         -> Settlement    (57% GT Settlement)
    5: 1,  # Tree         -> Forest        (75% GT Forest)
    6: 0,  # Water        -> ignored        [no counterpart in the target taxonomy; see below]
    7: 2,  # Agriculture  -> Grassland     [CHANGED from Cropland]   (82% GT Grassland; Irish ag = pasture)
    8: 4,  # Building     -> Settlement    (85% GT Settlement)
}


# ---------------------------------------------------------------------------
# Source classes with no counterpart in the target taxonomy.
#
# The six-class Biodiversity taxonomy has no water class, so OEM Water has nothing to map onto. The
# confusion still returns an argmax for it (Grassland, 0.553), but that measures teacher DOMAIN
# ERROR, not a correspondence: the matrix rows are the frozen teacher's PREDICTIONS on Irish
# imagery, whereas the map is applied to OpenEarthMap GROUND-TRUTH labels on OpenEarthMap imagery.
# Irish wet and shadowed grassland makes the teacher call grassland "water"; the inverse relabel
# would assert that genuine OEM lakes and rivers are grassland and inject that into pre-training.
#
# Such classes are EXCLUDED from the pre-training loss (mapped to Background, the training
# ignore_index) rather than forced onto the nearest class. This is the same treatment the
# harmonisation already gives the unmatched direction -- no OEM class corresponds to Cropland -- so
# it is one rule applied consistently to both, decided a priori from the class definitions rather
# than from any confidence value. Water is the only OEM class meeting it.
#
# NOT applied to Bareland (argmax Seminatural, 0.442) or Rangeland (Grassland, 0.495) despite their
# similarly weak argmax: both have genuine counterparts, and in the Irish uplands bare-looking
# ground really is rough semi-natural habitat. Bareland additionally supplies every semi-natural
# pixel in the relabelled pool, so excluding it would strip the priority class from pre-training.
# Weak confidence alone is therefore NOT the criterion; absence of a counterpart is.
# ---------------------------------------------------------------------------
OEM_NO_TARGET_COUNTERPART = {
    6: "Water — the target taxonomy has no water class; argmax Grassland (0.553) is teacher "
       "domain error on Irish wet/shadowed grassland, not a taxonomic correspondence.",
}
for _o in OEM_NO_TARGET_COUNTERPART:
    assert OEM_TO_STUDENT_PRETRAIN[_o] == 0, (
        f"OEM class {_o} ({OEM_NATIVE_CLASSES[_o]}) is declared to have no target counterpart, so "
        f"it must map to Background (the ignore_index), not {OEM_TO_STUDENT_PRETRAIN[_o]}"
    )


# The legacy name-based KD soft map (oem_to_student_kd) was REMOVED 2026-06-19.
# The campaign KD map is grounded in the teacher's empirical confusion:
# geoseg/utils/kd_utils.build_mapping_from_confusion("B"). Pre-training uses
# OEM_TO_STUDENT_PRETRAIN (above) = argmax of that same confusion.


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
NUM_STUDENT_CLASSES = len(STUDENT_CLASSES)   # 6
NUM_OEM_CLASSES = len(OEM_NATIVE_CLASSES)    # 9


def student_name(i: int) -> str:
    return STUDENT_CLASSES[i]


def oem_native_name(i: int) -> str:
    return OEM_NATIVE_CLASSES[i]
