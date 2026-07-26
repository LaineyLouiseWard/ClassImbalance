# Things the methods section must state, with the measurement behind each

Started 2026-07-26, **before any model was trained on the corrected split**. This is the working list
of properties of the design that a reader is entitled to know and that the results cannot be read
correctly without. Each entry gives the number and how it was obtained, so §2 can be written from
here rather than from memory.

Two rules for this file. Every number is recomputed from the shipped configs or from the tiles on
disk — never from a code comment, a note, or the withdrawn leaking campaign. And an entry is added
when the property is discovered, not when the results make it convenient.

Related records: `docs/DECISIONS_REBUILD_2026-07.md` (why each design decision was taken),
`docs/PREREGISTRATION_P1_AMENDMENT.md` (withdrawn 2026-07-26, retained with its history intact).

---

## 1. The transfer arm receives exactly twice the Biodiversity gradient steps

**Measured 2026-07-26, from the imported configs and the tiles on disk.**

Stage 2a pre-trains on a combined pool. Counted on `data/oem_combined_f1/train/images`: the pool is
3,190 tiles, of which **1,072 are the Biodiversity training tiles themselves** — pool ∩ train = 1,072,
train − pool = 0. Pre-training is therefore not a pass over a foreign dataset. It is a pass over
OpenEarthMap *and* a second pass over the training set.

| cell | tiles/epoch | steps/epoch (batch 2) | epochs | Biodiversity gradient steps |
|---|---|---|---|---|
| baseline (`stage1_baseline`) | 1,072 | 536 | 45 | 24,120 |
| pre-train (`stage2a_oem_pretrain`) | 3,190 | 1,595 | 45 | 24,120 — the 33.605% Bio share of 71,775 |
| finetune (`stage2b_oem_finetune`) | 1,072 | 536 | 45 | 24,120 |

**Baseline 24,120. Transfer arm 24,120 + 24,120 = 48,240, a ratio of exactly 2.00×.** Exact rather
than approximate: every Biodiversity training tile is seen once per epoch in both stage 2a and stage
2b, so the transfer arm passes over the training set 90 times against the baseline's 45. The same
holds on the other level of the sampler factor — `stage_sampler_only` 24,120 against `stage3_clsbal`
48,240 — so the confound sits squarely on main effect A.

**What must be written:** main effect A is not "cross-dataset transfer". It is *transfer plus a
second pass over the training set*, and this design cannot separate the two. A reader may not
attribute the transfer contrast to cross-dataset representation alone.

**Why it was not fixed rather than declared** (D12): the step-matched control would pre-train on
OpenEarthMap only, which removes the Biodiversity half of the pool and with it every Cropland and
Semi-natural label — see §3 below. That trades a declared confound for an undeclared one. Identified
and written down before any results existed, which is the only thing that makes a limitation
defensible rather than an excuse.

## 2. The transfer arm receives two checkpoint-selection passes to the baseline's one

**Measured 2026-07-26 by importing all five configs.** Every one monitors `val_mIoU` on the same
173-tile validation split with `save_top_k=1`. Stage 2a selects a best epoch on that split; stage 2b
then selects again from the model 2a handed over. Baseline and sampler-only select once.

Validation sits 256 m from training, and `report_class_support.py` marks all five of its foreground
classes weak. The usual defence — that validation optimism is common-mode across the cells — holds
for a level shift but not for a *selection rule*, and the arm that selects twice is the arm carrying
the positive result.

**What must be written:** the selection asymmetry, alongside §1. It is declared, not corrected.

## 3. OpenEarthMap contributes no Cropland and no Semi-natural labels

**Measured 2026-07-26 over all 3,190 pool masks.** Foreground shares within each half of the pool:

| | Forest | Grassland | Cropland | Settlement | Semi-natural |
|---|---|---|---|---|---|
| OpenEarthMap half (2,118 tiles) | 27.044% | 54.413% | **0.000%** | 18.542% | **0.000%** |
| Biodiversity half (1,072 tiles) | 14.579% | 70.111% | 7.703% | 3.342% | 4.265% |
| pool overall | 25.566% | 56.275% | 0.914% | 16.739% | 0.506% |

The grounded argmax maps Bareland, Rangeland and Agriculture all to Grassland, so no OpenEarthMap
class lands on Cropland or Semi-natural.

**What must be written:** for those two classes the transfer factor is *representation* transfer, not
label transfer.

**What must NOT be written:** that the two output channels are suppressed. They are not. Both receive
positive evidence throughout stage 2a from the Biodiversity half — Cropland 21,279,164 px across 248
tiles, Semi-natural 11,780,653 px across 261 tiles. What is real is a prior shift: 0.914% and 0.506%
of pool foreground against 7.703% and 4.265% in the target training set, about an eighth of target
prevalence. Correcting a source/target prior shift is what the stage-2b Biodiversity-only finetune
exists to do.
