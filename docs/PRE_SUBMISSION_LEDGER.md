# Pre-submission ledger — every finding from the 2026-07-26 verification

Eight independent verification agents, each finding adversarially attacked. 42 findings survived.
This is the complete list with nothing folded away: every one is mapped to a group and a status, so
"is it covered?" has an answer for each row rather than for the pile.

Status: **DONE** = fixed and verified by execution | **OPEN** = not yet | **N/A** = declared decision.

| # | finding | group | status |
|---|---|---|---|
| **A. Launch blockers** ||||
| A1 | `args.seed` undefined — every training run crashes before `trainer.fit` | A | DONE |
| A2 | seed worktrees never get `stseg_base.pth`; 36 of 40 runs cannot parse their config | A | DONE |
| **B. Before launch** ||||
| B1 | STAGES reorder inert: blocks execute in file order, gate still ran after 3 training runs | B | DONE |
| B2 | documented resume `--from B1` skips the sampler build and the only full gate | B | DONE |
| B3 | "all 432 parameter tensors identical across seeds" — false, measured in one process | B | DONE |
| **C. Withdrawn-campaign contamination and the evidence path** ||||
| C1 | stage E rebuilds the whole 13-figure manuscript set from the withdrawn split, reports 13/13 OK | C | DONE |
| C2 | stage E syncs five of those figures into `manuscript/Figures/` | C | DONE |
| C3 | `ablation_qualitative` figure built from the withdrawn campaign, exit 0 | C | DONE |
| C4 | `oem_mapping.tex` (main.tex:181) embeds the withdrawn confusion matrix; its generator crashes so drift is invisible | C | DONE |
| C5 | C3/C4 runbook stages accept a metrics.json of withdrawn provenance; only `aggregate_seeds` validates | C | OPEN |
| C6 | withdrawn deliverables sit at the exact output paths the pipeline writes, unmarked | C | OPEN |
| C7 | **the trimap exclusion curve — the primary evidence since D18 — is in no RUNBOOK stage**, and one input is produced by nothing | C | DONE |
| C8 | `accuracy_vs_separation.py` silently drops the entire external_test stratum | C | DONE |
| C9 | campaign dumps softmax for test/external only; three stage-E consumers read *val* dumps against the withdrawn 219-tile mask dir | C | DONE |
| C10 | `dump_seed_softmax.py` defaults to the withdrawn split; help text says 231 tiles | C | DONE |
| C11 | `boundary_rate_ratio.py --self-test` exercises neither the band definition, the tile-exclusion rule, nor the per-site distance conventions | C | DONE |
| **D. Gate and launcher hardening** ||||
| D1 | `external_test` is in no separation pair — Test B independence is asserted, never verified | D | DONE |
| D2 | the gate never reads `masks/`, so a split that has lost masks passes with wrong counts printed | D | DONE |
| D3 | a manifest waives its own adequacy check by omitting the keys; nothing downstream restores it | D | DONE |
| D4 | ~~the A0 taxonomy gate prints PASS when the name it guards is deleted~~ **DID NOT REPRODUCE** — the gate fails, exit 1, 2 of 31 checks. The `getattr` default was removed anyway, since a deletion plus a matching dict edit would have passed | D | DONE |
| D5 | B5's `require_file` gates on stage2b, but `stage3_clsbal` initialises from stage2a | D | DONE |
| D6 | the clean-tree check runs on the submitting checkout; array tasks run from unverified `$SONIC_SCRATCH/seed<N>` | D | DONE |
| **E. False statements in tracked text** ||||
| E1 | D2's "18% of tiles lost to buffers" — measured 19.3% of the pool, 21.2% of the cut site | E | DONE |
| E2 | D14 asserts "950 m on the full pool", which METHODS §4 explicitly forbids and no artefact supports | E | DONE |
| E3 | CLAUDE.md's "registered claim" section contradicts D18; `boundary_rate_ratio.py` still prints a registered verdict | E | DONE |
| E4 | `utils.spatial_blocks` docstring says "~14 independent units at 950 m"; the function returns 16 | E | DONE |
| E5 | `stage3_clsbal.py`: "num_samples = len(train set) = 1846 ... the same 1846 Bio tiles" — it is 1072 | E | DONE |
| E6 | `taxonomy.py`: OEM Water "argmax Grassland (0.553)" — the shipped matrix says 0.7525 soft / 0.7485 hard | E | DONE |
| E7 | shipped factorial configs carry stale comments and a copy-pasted docstring misdescribing the cells | E | DONE |
| E8 | METHODS §6 changes what "block" means mid-paragraph; the boundary-free exclusion costs zero blocks | E | OPEN |
| E9 | METHODS §7's interval numbers have no committed code, contradicting the file's own opening rule | E | OPEN |
| E10 | `artifacts/replication_exposure_report.json` is a month stale, matches no split, referenced by nothing | E | OPEN |

## Not defects — declared decisions, recorded with evidence

Re-reported by reviewers without the decision log; each stays as it is.

- the transfer arm's 2.00x gradient-step confound and second val-selection pass — **D12**
- no rho threshold, no dead band, no weak band — **D18**
- 950 m being ireland2's composition range rather than the inland site's — **D14**, justified in METHODS §4
- the split clearing its adequacy floors at 5 of 10 grid phases — declared, METHODS §5
- `README.md`, `RUNBOOK.md`, `docs/DESIGN_NOTES.md` describing the old split — stale by decision, fixed after the campaign

## What none of the 42 touched

The split geometry, the buffers, the two estimands, the block-support criterion that admitted the
split, and the factorial contrasts. Those were attacked across four review rounds and held every
time. Every finding above sits in the plumbing between training and reporting, in a gate, or in
prose.
