# What is in `docs/`, and what to read first

A campaign was **withdrawn on 2026-07-25** for train/test leakage, and the split was rebuilt. That
means this directory contains two very different kinds of document, and reading them in the wrong
order gives entirely the wrong impression of the project. **Read the first group. The second group is
the record of how the first group was arrived at.**

## Read these — they state the design as it now is

| file | what it is |
|---|---|
| **`METHODOLOGICAL_CHOICES.md`** | every deliberate choice in plain language, with what it costs. **Start here.** |
| **`METHODS_STATED_LIMITATIONS.md`** | eleven properties the methods section must state, each with the measurement behind it |
| **`CORRECTIONS.md`** | sentences currently in the manuscript that are wrong and must change |
| **`DO_NOT_ADD.md`** | sentences the design forbids, and sources that do not exist or do not say what they are cited for |
| **`FIGURES.md`** | the figure map |

## This is the audit trail, not the design

These record a withdrawn campaign and the rebuild that followed. They are kept, tracked and
unedited **on purpose**, and that is the point of them:

`DECISIONS_REBUILD_2026-07.md` (D1–D19) · `PREREGISTRATION_P1_AMENDMENT.md` (three versions, two
retracted) · `PRE_SUBMISSION_LEDGER.md` · `AUDIT_7_FINDINGS.md` ·
`METHODOLOGY_REVIEW_2026-07-27.md` · `PLAN_FINAL_CATCHALL_AUDIT.md` · the `BRIEF_*.md` files

**Why they look like they do.** A leakage retraction guarantees somebody asks: *were the second set
of numbers tuned after the first set was withdrawn?* The honest answer needs dates, and this project
has them. Every decision in `DECISIONS_REBUILD_2026-07.md`, and all three versions of the
pre-registration including the two retractions, are dated **before any model was trained on the
corrected split.** A statistic withdrawn before you have seen a single result cannot have been
withdrawn because you did not like the result.

So the 23 supersede/reverse/withdraw markers in the decisions log are not churn to be embarrassed
about — they are the audit trail doing its job, in the one situation where a project most needs one.
Note also what the reversals converged *on*: the removal of machinery (a threshold, a block
bootstrap, a coverage simulation), not the addition of it.

## Stale by decision

`README.md`, `RUNBOOK.md` and `DESIGN_NOTES.md` still describe the withdrawn 1,706/219/218 split and
state its conclusions as fact. They now carry banners saying so and are rewritten after the rebuilt
campaign runs. Do not read a number out of any of them.
