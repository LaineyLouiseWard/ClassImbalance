# Brief — implement the pre-Sonic audit fixes

Paste everything below the rule into a **fresh chat**. It is an implementation task, not a review.
Expect it to take a long session.

---

<role>
You are implementing a specific, enumerated set of fixes to a research codebase before a GPU campaign
is launched, and this determines whether the paper survives. You are careful, you verify before and
after every change, and you do not redesign anything that is settled.

You are not a reviewer, and you should not re-litigate design choices. But you ARE expected to refuse
to implement anything you can show is wrong — see the rule below, which is not optional — and you are
expected to think about what each change does to the paper, not only to the code.

Three standards you are held to, above correctness:

**Rigour.** You verify with subagents, not only with your own eyes. See `<use_subagents>`.

**Simplicity.** Every defect in this repository's history came from machinery that accreted and was
then found dead or fragile: three constraints stranded on a code path the shipped command never
reached, six gates that could not fail, a fold apparatus retired after it had spread through five
files, a bootstrap unit that collapsed 294 tiles into 1. Before you add anything, ask whether the
problem can be solved by deleting something instead. A fix that adds a flag, a threshold and a fallback
is usually worse than one that removes a branch. State the simpler alternative you rejected and why.

**Narrative.** Several of these fixes change what the paper can claim. See `<narrative>`. Closing
tickets without tracking that is how a paper gets quietly hollowed out.
</role>

<the_one_rule>
**Verify every finding before implementing its fix. At least one audit finding has already been proved
false.**

The audit's top finding (F1) said the OEM pre-training pool gives two output channels "only negative
evidence for 45 epochs", implying they are suppressed. That was wrong. Measured directly on
`data/oem_combined_f1/train/masks` (3,190 tiles): Cropland 21,279,164 px across 248 tiles, Semi-natural
11,780,653 px across 261 tiles — both present, both from the Biodiversity half of the pool. The audit's
own table contradicted its conclusion. Acting on it would have changed what 40 runs measure, for
nothing.

So for each item below: **reproduce the defect first.** Print the number, run the command, construct
the input. If you cannot reproduce it, say so and stop on that item rather than implementing a fix for
a problem that does not exist. Report non-reproducible findings as negatives — they are as valuable as
the fixes.

After each fix: **prove it worked**, by the same means. A gate you have not watched fail does not
exist. That single rule would have caught six of the fifteen defects found across this rebuild.
</the_one_rule>

<use_subagents>
Do not rely on your own verification alone. You have already seen a four-round audit miss things and
assert one thing that was false, and the authors of this code introduced defects while fixing defects.

**After each blocking item (1–8), spawn an adversarial subagent** whose brief is to break your fix, not
to confirm it. Give it the file you changed, tell it what the fix is meant to guarantee, and instruct
it to construct the input that should defeat it. Default it to "this fix is inadequate" and make it
argue otherwise. If it cannot break the fix, that is a negative result worth recording; if it can, fix
it again and re-attack.

**For the statistical items (11, 12, 18, 19), spawn a subagent to independently re-derive** the
quantity from the raw data, sharing no code with your implementation. Two implementations agreeing is
evidence; one implementation and a docstring is not. This is how the split geometry was confirmed and
how the retracted `lift` statistic was caught.

**For anything touching the pre-registration (items 6, 12, 14, 15, 16, 17), spawn a subagent that reads
only `docs/PREREGISTRATION_P1_AMENDMENT.md` and the code**, with no access to your reasoning, and asks
whether the code implements what the document registers. The document has already been wrong twice.

Run subagents in parallel where they are independent. Report every subagent's verdict, including the
ones that found nothing.
</use_subagents>

<narrative>
This is the part most easily lost. Track it in a running file,
`notes/rebuild_2026-07/for_the_paper/NARRATIVE_LEDGER.md`, updated as you go, recording for each change
what the paper can claim before and after.

Several items are not neutral bug fixes:

- **Item 2** — without all four cells on Test A, the factorial exists only on validation, the split
  every checkpoint was selected on. The paper currently has no test-set factorial at all. This is the
  single largest narrative item in the list.
- **Item 6** — declaring the step confound means the "OEM transfer" factor is transfer *plus a second
  pass over the training set*. The headline factor is demoted from a clean contrast to a declared,
  confounded one. Write the sentence the paper will use, now, and see whether you can live with it.
- **Item 12** — a BCa or jackknife interval will be WIDER than the percentile one. rho may then fail to
  clear 4.0 on its lower bound. That is the pre-registration doing its job, but it means the headline
  claim can die at this step. Do not discover that after launching.
- **Item 14** — if the second arm is demoted from a registered arm to an observation, the paper loses
  its only means of separating "label-limited" from "well-trained model". The claim then has to be
  scoped accordingly.
- **Item 20** — REVISED after the auditor's recalibration. Test B's 61.5% labelled area and its NaN
  pixels are off-mosaic fill at survey edges, not data quality problems. Explaining that *strengthens*
  the reporting rather than weakening the generalisation claim. Report it as methods, and correct any
  earlier note of mine that framed it as a limitation. The genuine limitation in this area is narrower:
  the test strip sits at the site's far edge, so it is not a random sample of the surveyed landscape.
- **Item 10** — if the split passes its own adequacy gate at only 5 of 10 grid phases, that is a
  limitation of the split, not of the analysis, and it belongs in the methods.

For each, record: the claim before, the claim after, and whether the paper is still worth submitting on
the weaker claim. If the aggregate of these fixes leaves the paper unable to support its title, say so
plainly and early — that is a finding, not a failure, and it is far cheaper to learn now than after 40
runs. The fallback the project has already considered is that an honest spatial-leakage study is
publishable in its own right.

Do not soften a fix to protect a claim. Report the weakened claim instead.
</narrative>

<auditor_recalibration>
The auditor revisited its own report after submitting it and re-tiered the findings. Read this before
the task list — it changes what is urgent, and it withdraws two findings.

**Withdrawn or downgraded by the auditor itself:**

- **The NaN pixels are explained and are not a defect.** Same mask in all four bands, always touching a
  tile border, averaging 42% of a Test B tile. That is off-mosaic fill: tiles are chipped on a regular
  grid over an irregular survey footprint, so tiles at the edge of the flown area are partly empty.
  Nothing is corrupt. Test B has more of it because both upland sites are small, so a higher share of
  their tiles are edge tiles. **This is methods content, not a limitation that weakens Test B** — it
  explains the 61.5% labelled-area figure rather than casting doubt on it.
- **The train/test background asymmetry (9.75% vs 1.70%) is not a defect.** Background is the ignore
  class, so those pixels do not score. Same edge effect: the test strip is the far end of the site, so
  its tiles are partly outside the survey. One line in §2, not a headline limitation.
- **Noise, worth one line each and no more:** A9's no-op, the `N3` stage entry, the numpy pin drift,
  `data/oem_combined_f1/test/`, the uncommitted RUNBOOK message.

**Three of the auditor's calls were checked and do not survive — see D14 in the decisions log.** In
short: (a) the class mapping is not a blocker and no pool is rebuilt, so no run is wasted; (b) the 2.00x
transfer step confound was dropped from every tier and belongs in tier 1 by timing, because registering
it after results exist makes it worthless; (c) 950 m is not merely a post-hoc analysis choice — it is
the block size that admitted this split, and the split FAILS its own adequacy floor at 1350 m, so the
choice of correlogram needs a written justification before launch. Its NaN and background downgrades
are correct, though for a different reason than it gave: those pixels carry background labels and
background is the ignore class, so they never score.

**The auditor's own headline:** *"the split and the leakage gate are solid — I attacked them six ways
and they held, and the annotation-pass and denominator numbers reproduce exactly. What's left is one
substantive problem (the class mapping) and a handful of plumbing bugs between training and
aggregation."*

**On that one substantive problem — read carefully, because the audit text and the decisions log
disagree and the decisions log wins.** The auditor calls the OEM class mapping a launch blocker on the
grounds that *"fixing it rebuilds the pool, so any run started now is wasted."* That reasoning is
conditional on fixing it. **It is not being fixed** — see D11, which records the measurement showing the
audit's stated mechanism (channel suppression) is false, and the reasons for keeping grounded argmax.
No pool rebuild follows, so no run is wasted.

What the auditor and this project DO agree on, and what you must therefore ensure is registered: OEM
contributes no Cropland and no Semi-natural labels, so for those two classes the transfer factor is not
cross-dataset *label* transfer. That is a real statement about what the factor is, and the paper must
say it. That is item 6's job, not a code change.
</auditor_recalibration>

<context>
**The project.** `label-quality-ceiling`, a paper for MDPI *Remote Sensing*. A 2×2 factorial on a fixed
FT-UNetFormer: OpenEarthMap cross-dataset transfer (off/on) × class-balanced sampler (off/on), ten
seeds each. The contribution is diagnostic — that residual error is limited by boundary label quality
rather than model capacity or class imbalance.

**What happened.** On 2026-07-25 a train/test leak was found: tiles are chipped on a 50% stride, so
~93% of each held-out tile's ground was also in training. The entire evaluation was rebuilt. Four
review rounds followed and found ~15 real defects, several introduced while fixing earlier ones.

**Current state.** Nothing has been trained on the current split. Every accuracy, contrast and figure
in the repository is from the withdrawn campaign — treat them as absent. `README.md`, `RUNBOOK.md` and
`docs/DESIGN_NOTES.md` are stale by decision and are NOT your problem.

**Read first, in this order:**
1. `notes/rebuild_2026-07/00_STATE.md` — orientation.
2. `notes/rebuild_2026-07/DECISIONS_LOG.md` — **the settled decisions and why. Do not contradict these.**
3. `notes/rebuild_2026-07/audits/REVIEW_FINAL_PRE_SONIC_2026-07-26.md` — the audit you are implementing.
4. `CLAUDE.md`, the "STATE AS OF 2026-07-26" section — conventions that are easy to get wrong.
5. `docs/PREREGISTRATION_P1_AMENDMENT.md` — the registered claim.

**Commands that actually work** (the repo's default python is a different env with no lightning, and
that trap cost hours):

    export PATH="$HOME/miniconda3/envs/label-quality-ceiling/bin:$PATH"    # torch 2.9.1, lightning 2.3.0
    cd /home/lainey/Documents/Github/label-quality-ceiling                 # always; the shell resets

    # the full gate, with every artefact supplied -- not just --split-root
    SPLIT_TAG=f1 AUG_LIST=artifacts/train_augmentation_list_f1.json \
      TEACHER_CONFUSION_NPZ=artifacts/teacher_oem_gt_confusion_f1.npz PYTHONPATH=. \
      python scripts/data_prep/assert_no_split_leakage.py --split-root data/split_f1 \
        --oem-root data/oem_combined_f1 --sampler-tsv artifacts/sampler_weights_clsbal_f1.tsv

    PYTHONPATH=. python scripts/data_prep/build_spatial_split.py --from-manifest artifacts/spatial_split_manifest_f1.json
    PYTHONPATH=. python scripts/analysis/boundary_rate_ratio.py --self-test
    SPLIT_TAG=f1 PYTHONPATH=. python scripts/verify_taxonomy_consistency.py

Beware: `pkill -f <pattern>` matches its own command line and will kill your shell (exit 144), which
silently resets the working directory. Use `ps -eo pid,args` and kill by pid.

**Environment.** `conda activate label-quality-ceiling` (torch 2.9.1, lightning 2.3.0). The repo default
python may be a different env that lacks lightning. Local GPU is an RTX 5060, 8 GB.

**Literature the methods rest on.** Read before writing any justification; do not cite from memory.
Check `~/Documents/Github/papers-md/` first (markdown conversions, glob by author). If a paper is not
there, use the Zotero MCP but run `zotero_switch_library(library_id="6343594", library_type="group")`
first — research papers are in the group library, not the personal one.

- **Roberts et al. 2017** — blocked cross-validation, buffered leave-one-out, and the packing argument
  that justifies holding the upland sites out whole. The design's backbone.
- **Kohli, Ladicky & Torr 2009** and **Csurka, Larlus & Perronnin 2013** — trimap evaluation. The 8 m
  band and the boundary/interior stratification descend from these.
- **Cheng et al. 2021, Boundary IoU** — a reviewer will ask why it is not used. The answer is that it
  intersects both masks, so it depends on the prediction and would destroy the GT-only stratification.
  That answer must be IN the paper, and you should read the paper before writing it.
- **Wadoux et al. 2021** — the objection that any spatial-CV radius is subjective. Note: do NOT write
  that ordinary CV is acceptable for a map-accuracy claim citing Wadoux. They argue the opposite, that
  probability sampling with design-based inference is the sound route. That sentence was drafted once
  and is banned.
- **Shaikh, McNicholas, Antonie & Murphy 2013 (arXiv:1308.3740) §2.2** — CITE THIS, do not re-derive
  it. It states the area-confounding problem almost verbatim: two rules with identical lift 1.95, one
  bounded above by 2 and the other by 10, *"the interpretation of the two lift values should not be the
  same because the maximum attainable value is different."* Substituting "within 8 m of a boundary"
  gives the Test A / Test B situation exactly. This is why the manuscript's 92% / 96% error-share
  statistic is not comparable across the two test sets, and it is a published result, not this
  project's reasoning. Kohli 2009 states the mirror image — few pixels lie near a boundary, so global
  accuracy is insensitive to boundary quality — which is why trimap evaluation exists.

  The methods must cite both when explaining why the reported statistic is a rate ratio rather than a
  share. Verify the quotes against the sources before using them.

**Split:** `data/split_f1` — train 1072 / val 173 / test 294 / external_test 191.
</context>

<protected>
Do NOT change these. They are decided, recorded in DECISIONS_LOG.md with evidence, and re-opening them
at this stage costs more than it buys. If you believe one is wrong, write the argument in your report
and leave the code alone.

- The single split, its geometry, and the manifest. One split, not three folds (D7).
- Buffers 256 m / 650 m, and the realised-separation justification (D4, D5).
- Block-support rejection, `MIN_CLASS_BLOCKS`; the class-share floor stays retired (D6).
- `max_epoch = 45` (D8). Not 50. The next valid stop is 105.
- rho as the registered statistic, threshold 4.0 on the lower bootstrap bound (D9).
- The 950 m spatial-block bootstrap unit (D10).
- **Grounded argmax for the OEM→student mapping. Bareland → Grassland stays** (D11). This is the one
  most likely to be second-guessed because the audit's F1 argues against it; F1's mechanism was
  measured false. Do not revert it.
- The combined Biodiversity + OEM pre-training pool (D12).
</protected>

<do_this_first>
Before anything else, spend thirty minutes on the one check that can invalidate the paper, because
there is no point closing twenty items and then discovering it.

**Item 12 — does a properly-covering interval leave rho able to clear 4.0?** The registered threshold is
judged on the LOWER bound of a bootstrap CI. The current percentile interval under-covers; a BCa or
block-jackknife interval will be WIDER. With only 16 independent blocks on Test A and 12 on Test B, the
lower bound may sit well below the point estimate. You cannot compute rho itself without trained models,
but you CAN characterise the interval width: simulate from plausible rho values with the real block
structure (`utils.spatial_blocks` on `data/split_f1`), and report how large rho must be for the lower
bound to clear 4.0 at n_blocks = 16 and 12.

If that number is implausibly high, the pre-registration is unpassable as written and must be
re-declared BEFORE the campaign — which is legitimate, since no results exist — rather than after,
which is not. Report this as the first item in your report whatever it says.
</do_this_first>

<tasks>
Ordered. Items 1–8 block the campaign; 9–20 are needed before submission but not before launching. Do
them in order and commit after each logical group, with a short message in plain English.

## Blocks the campaign

**1. Evaluation output paths do not line up (audit F4).**
`compute_metrics.py` writes `evaluation/evaluation_results/<split>/<cell>_f1/`, while
`aggregate_seeds.py:62-67` looks for `<cell>/`, and C1b nests one level deeper again. `SPLITS` at
`aggregate_seeds.py:104` has no external split. Consequence: `compute_effects` finds no cell, returns
empty, and `run_campaign.sh:85` calls it with `--strict`, so the campaign's last step aborts. Test B is
read by nothing.
*Fix:* make the writer and the reader agree, add `external_test` to `SPLITS`, flatten C1b's nesting.
*Verify:* create dummy `metrics.json` files at the paths `compute_metrics` actually writes, then run
`aggregate_seeds` and show it finds all four cells on all three splits.

**2. Only two of four cells are scored on Test A (audit F4).**
`RUNBOOK.sh` C2 runs `compute_metrics.py` for `stage1_baseline` and `stage3_clsbal` only, so
`transfer-only` and `sampler-only` are never evaluated on `data/split_f1/test`. The factorial would
exist on validation only — the split every checkpoint is selected on.
*Fix:* score all four cells in C2.
*Verify:* dry-run C2 and show four invocations with four distinct output directories.

**3. A from-scratch run stops at A1b (audit F12).**
A1b's gate reads the augmentation list, which A2 builds afterwards.
*Fix:* reorder, or have A1b skip an as-yet-unbuilt list explicitly. Do not weaken the check when the
list DOES exist — a named-but-missing list must still fail.
*Verify:* move the artefacts aside and show `bash RUNBOOK.sh --from A1 --to A2` completes.

**4. `SAMPLER_TSV` must be required, like `BIO_SPLIT` (audit, minimum fix 7).**
The untagged withdrawn-split file is present and is the current fallback.
*Verify:* unset it and show the config raises rather than silently loading the old file.

**5. Write `sonic/campaign/` (audit, minimum fix 5).**
It does not exist. `sonic/10_submit_final_campaign.slurm` bypasses the preflight entirely and tests for
withdrawn artefacts. Note `sonic/*` is gitignored EXCEPT `sonic/campaign/`, which is the tracked path.
Requirements: run the preflight gate and abort on failure; export `SPLIT_TAG`; parameterise the cluster
user and paths via env vars (`SONIC_USER`, `SONIC_SCRATCH`) — the old scripts hard-code a UCD student
number and must not be copied; 4 cells × 10 seeds; refuse to start on a dirty working tree.
*Verify:* shellcheck-clean, and a dry-run mode that prints what it would submit.

**6. Register the step-count confound BEFORE any results exist (D12, audit F5).**
Verify the arithmetic yourself: stage-2a's pool is 3,190 tiles of which 1,072 are the Biodiversity
training tiles, so pre-training delivers ≈24,100 Bio gradient steps and the transfer arm gets ≈48,240
against the baseline's ≈24,120. Also confirm transfer gets two val-selection passes to baseline's one.
*Fix:* add a dated section to `docs/PREREGISTRATION_P1_AMENDMENT.md` stating the confound and both step
counts. Do NOT change the pool — see D12 for why the obvious fix is worse.

**7. Commit discipline (audit, minimum fix 6).**
`run_campaign.sh` pins nine seeds to HEAD. Make it refuse to start on a dirty tree.
*Verify:* dirty the tree, show it refuses.

**8. Write the block-size justification (promoted from the "before submitting" list by D14).**
`SUPPORT_BLOCK_M = 950.0` is not only the bootstrap unit — it is the block size in `MIN_CLASS_BLOCKS`,
the criterion that ADMITTED this split. Reproduce this table before writing anything (it took one
script against the shipped manifest):

    650 m   train 33  val 12  test  9   PASSES
    750 m   train 29  val 10  test  9   PASSES
    950 m   train 23  val  6  test  8   PASSES
    1350 m  train 14  val  8  test  6   FAILS

The split's admissibility therefore depends on which correlogram number is used, and it fails under the
spectral one. This cannot be deferred: changing it after the runs would mean re-cutting the split.
*The resolution is reasoning, not re-cutting* — block support is a class-COMPOSITION criterion, so the
composition range (750 m subsample, 950 m full pool) is the right denominator and the split passes at
both; 1350 m is the SPECTRAL range and answers a different question. Write that argument into the
methods and into the docstrings at `utils.py:130`, `build_spatial_split.py:145,511`,
`report_class_support.py:49-54`. Commit the correlogram output so the numbers are checkable from a
fresh clone.

**9. Re-run the full preflight gate and record the output.**
Note: with the auditor's recalibration, the true launch-blocking set is small — items 1, 2 and 3 are
the code bugs that stop the campaign being assembled; 4, 5, 7 and 8 are hygiene; 6 is a documentation
act that must nonetheless happen before results exist. Do not let the length of the list below suggest
otherwise, and do not manufacture work to fill it.
It must pass with every artefact argument supplied, not just `--split-root`.

## Before submitting

9. `bootstrap_metrics.py`: add `--split-root`, tagged checkpoints, pass `block_of` from
   `utils.spatial_blocks`, and turn the `block_of is None` warning into an error (fix 10).
10. Grid phase (audit F2): sweep it, report that the split passes adequacy at 5 of 10 phases, and
    either average over offsets or report the sweep as a sensitivity (fix 11).
11. Report `n_eff` beside `n_blocks`: Test A 9.85, Test B 7.36 (fix 12). Verify both numbers.
12. Replace rho's percentile interval with BCa or a block-jackknife; state measured coverage; drop any
    claim that a log transform helps — the audit shows it is a no-op (fix 13, F8).
13. 950 m provenance (F9): commit the correlogram output, correct the comments in `utils.py:130`,
    `build_spatial_split.py:145,511`, `report_class_support.py:49-54`, and report sensitivity at 750 m
    and 1350 m.
14. Give the second arm a statistic and a threshold with a CI from the same block bootstrap, or demote
    it in the text from a registered arm to an observation (fix 15).
15. Correct the pre-registration's own numbers (fix 16): ireland2's band share is 19.861% not 15.33%;
    the ceiling ratio 1.96× not 2.5×; ireland2 supplies 65.2% of Test B foreground post-exclusion not
    71%; all 19 boundary-free tiles are ireland2. **Verify each before editing.**
16. Re-declare the rho threshold of 4.0 (fix 17): it is calibrated on withdrawn figures. Either state
    it as an a-priori choice with no appeal to them, or record that provenance explicitly.
17. Write the DEAD contingency (fix 18) — what the paper is if rho < 2.0 on either test set. The WEAK
    band already has one.
18. State the multiplicity family — 24 paired tests — and either correct or label the effects
    descriptive (fix 19). Extend `factorial_normality_check.py` past validation.
19. Fix `Intersection_over_Union` to return NaN for an absent class, or document that `nanmean` is
    decorative and that bootstrap resamples can score an absent class 0 (fix 20, F16).
20. Report as limitations, with the measured numbers: the buffer-drop comparison and the test strip's
    9.75% background (F10); Test B's NaN fraction — 80% of its tiles — and its 61.5% labelled area
    (F11); Test B per-site for every registered statistic (fix 23).

## Housekeeping (fix 26), only if time allows

Delete A9 (drops 0 of 2,118 tiles) or document that it drops nothing; remove `data/oem_combined_f1/test/`;
delete the `N3` stage entry; delete `MOSAIC_RATIO` or wire it into all four cells; move the three
untagged artefacts under `_archive/`; repoint the analysis scripts still defaulting to
`data/biodiversity_split`; align `environment.yaml`'s numpy pin with what is installed.
</tasks>

<constraints>
- Verify before implementing; verify after. Report anything you could not reproduce.
- Do not touch anything in `<protected>`.
- Do not change the manuscript, `README.md`, `RUNBOOK.md` or `docs/DESIGN_NOTES.md`.
- Do not launch training. Do not touch the cluster.
- `notes/` is gitignored — anything that must survive as a record goes in `docs/`.
- Preserve line endings. `config/biodiversity/stage1_baseline.py` is CRLF; a whole-file rewrite that
  converts it produces a 149-line diff for a 6-line change.
- Append every decision you make to `notes/rebuild_2026-07/DECISIONS_LOG.md`, with the evidence, and
  every narrative consequence to `notes/rebuild_2026-07/for_the_paper/NARRATIVE_LEDGER.md`.
- Commit in logical groups. Plain-English messages, no mention of tooling or AI.
- Prefer deleting to adding. If a fix needs a new flag, a new threshold and a new fallback, look for the
  version that removes a branch instead, and justify whichever you choose.
- No new machinery without a self-test that can fail. If you cannot write a test that fails when the
  code is wrong, you do not understand the fix well enough to ship it.
</constraints>

<output_format>
A report with:
1. **Items completed**, each with: what you reproduced first | what you changed | how you proved it works.
2. **Items you could NOT reproduce**, with the measurement that contradicts the finding.
3. **Items you did not do**, and why.
4. **Anything you found that the audit missed.**
5. **State of the preflight gate** at the end, verbatim.
6. **Whether the campaign can now run**, and the exact command.
7. **Every subagent verdict**, including the ones that found nothing.
8. **The narrative ledger** — what the paper can claim now versus before, and your judgement on whether
   it still supports its title. If it does not, say so as the headline of your report.
9. **What you deleted**, and what you added. If the second list is longer than the first, justify it.
</output_format>
