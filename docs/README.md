# What is in `docs/`

This directory documents the design, results and methodology behind the code, plus a dated integrity
trail from an early withdrawn campaign.

An earlier campaign was **withdrawn on 2026-07-25** for train/test leakage: tiles were chipped on a
50% stride and split at random by tile, so most of each held-out tile's ground was also in training.
The split was rebuilt as a spatially blocked split and every result regenerated. The manuscript
reports the rebuilt results.

## The design and results

| file | what it is |
|---|---|
| **`METHODOLOGICAL_CHOICES.md`** | every deliberate design choice in plain language, with what it costs. **Start here.** |
| **`RESULTS_TEN_SEED.md`** | the numbers (§7–§13 are authoritative) |
| **`FINDING_BOUNDARY_IS_PER_CLASS.md`** | the per-class boundary result and its confounds |
| **`METHODS_STATED_LIMITATIONS.md`** | properties the methods section states, each with the measurement behind it |
| **`DESIGN_NOTES.md`** | design decisions and the negative results (knowledge distillation, the bespoke sampler) |
| **`CORRECTIONS.md`** | corrections applied during manuscript preparation |
| **`DO_NOT_ADD.md`** | claims the design forbids, and sources that do not say what they might be cited for |
| **`FIGURES.md`** | the figure map |
| **`NUMBERS.md`** | every number the write-up quotes, the file that holds it, and the command that rebuilds it |

Before quoting any number, run `PYTHONPATH=. python scripts/analysis/verify_narrative_numbers.py` — it
checks each figure the narrative uses against the committed artifact behind it, and prints the
regenerating command for anything missing.

## The integrity trail

`audit/DECISIONS_REBUILD_2026-07.md` (the D1–D19 decisions log) and
`audit/PREREGISTRATION_P1_AMENDMENT.md` (three versions, two retracted) are kept, tracked and dated
**on purpose**.

A leakage retraction guarantees somebody asks: *were the second set of numbers tuned after the first
was withdrawn?* The honest answer needs dates, and this project has them. Every decision in the log,
and all three versions of the pre-registration including the two retractions, are dated **before any
model was trained on the corrected split**. A statistic withdrawn before a single result was seen
cannot have been withdrawn because the result was unwelcome. The reversals also converged on the
*removal* of machinery (a threshold, a block bootstrap, a coverage simulation), not its addition.
