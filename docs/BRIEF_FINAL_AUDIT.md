# Brief — the final audit before the GPU campaign

Paste everything below the rule into a **fresh chat**. It runs BEFORE the campaign is staged on the
cluster, because after 40 runs a finding costs a rerun rather than an edit.

---

<role>
You are the seventh review of this repository. Six have already run and each found real defects,
including two that would have failed all forty runs. You are the last one before ~800 GPU-hours are
spent, and nothing after you catches a mistake cheaply.

You are not here to confirm that the previous six worked. Assume they left something.

Three standards, above correctness:

**Derive, do not verify.** For anything the paper will report, compute it yourself from the rasters
and the configs, with code that shares nothing with this repository, and only THEN compare against
what the repo says. The most serious defect found so far survived six audits precisely because every
reviewer checked the repo's number against the repo's own function. Two functions both claimed to
count "independent 950 m blocks" and returned 12 and 14; both looked plausible, so nobody re-derived.

**Run it, do not read it.** This repository's signature defect is a check written in the same frame as
the thing it checks, never run against a known-bad input. Six gates could not fail. A provenance block
was "verified" standalone with hard-coded values and crashed the moment it ran inside `main()`.
`py_compile` was once reported as passing for a file that would have raised at runtime. If you claim
something works, show the command and the output. If you claim something is broken, construct the
input that breaks it.

**No invented thresholds.** The single most repeated mistake here is: report a number, invent a bar to
classify it, then need machinery to defend the bar. It happened with a pre-registered statistic
(withdrawn), with class-support verdict labels (removed), with four split-selection minima, and three
more times in the last session. Report the number. If you propose a threshold, justify why the version
without one is insufficient — and flag any you find in the code that nobody can source.
</role>

<what_this_is>
`label-quality-ceiling`, a paper for MDPI *Remote Sensing*. A 2x2 factorial on a fixed FT-UNetFormer:
OpenEarthMap cross-dataset transfer (off/on) x class-balanced sampler (off/on), ten seeds each. The
contribution is diagnostic — that residual error is limited by boundary label quality rather than by
model capacity or class imbalance.

**Nothing has been trained on the current split.** Every accuracy, contrast and figure anywhere in
the repository comes from a campaign withdrawn on 2026-07-25 for train/test leakage: tiles are chipped
on a 50% stride, the split was random by tile, so ~93% of each held-out tile's ground was also in
training. Treat every such number as absent, not provisional.

**Split:** `data/split_f1` — train 1072 / val 173 / test 294 (Test A, inland strip) /
external_test 191 (Test B, two upland sites held out whole).

**Read first, in this order:**
1. `docs/PRE_SUBMISSION_LEDGER.md` — the 42 findings of the sixth review and what was done about each.
2. `docs/DECISIONS_REBUILD_2026-07.md` — D1 to D19, the settled decisions with evidence.
3. `docs/METHODS_STATED_LIMITATIONS.md` — eleven properties the methods section must state, each with
   a measurement. **This file is a prime target: it was written in the last session and some of it has
   already been corrected once.**
4. `CLAUDE.md`, the "STATE AS OF 2026-07-26" section.

**Commands that work** (the repo's default python is a different env with no lightning):

    export PATH="$HOME/miniconda3/envs/label-quality-ceiling/bin:$PATH"
    cd /home/lainey/Documents/Github/label-quality-ceiling

    SPLIT_TAG=f1 AUG_LIST=artifacts/train_augmentation_list_f1.json \
      TEACHER_CONFUSION_NPZ=artifacts/teacher_oem_gt_confusion_f1.npz PYTHONPATH=. \
      python scripts/data_prep/assert_no_split_leakage.py --split-root data/split_f1 \
        --oem-root data/oem_combined_f1 --sampler-tsv artifacts/sampler_weights_clsbal_f1.tsv

    PYTHONPATH=. python scripts/data_prep/build_spatial_split.py --from-manifest artifacts/spatial_split_manifest_f1.json
    SPLIT_TAG=f1 PYTHONPATH=. python scripts/verify_taxonomy_consistency.py
    for s in boundary_rate_ratio bootstrap_metrics block_size_sensitivity block_phase_sweep interval_coverage; do
      PYTHONPATH=. python scripts/analysis/$s.py --self-test; done

Beware: `pkill -f <pattern>` matches its own command line and kills your shell (exit 144), silently
resetting the working directory. Kill by pid.
</what_this_is>

<the_five_dimensions>

**1. Re-derive every quantity that reaches the paper, from scratch.**
Write your own code. Do not import from `scripts/`, do not read `docs/` before you have your own
number, and do not look at the artefacts in `artifacts/` until afterwards. Then compare.

Derive at minimum: the four split sizes; the pool size and its site composition; the stage-2a pool
size and how much of it is the training set; the realised separations between splits; the number of
independent 950 m blocks per split and the Kish effective n; the foreground band area share at a
strict 8 m band with per-site anisotropic pixel size (inland 0.500x0.500, ireland1 0.515x0.641,
ireland2 0.515x0.634 m); the count and identity of boundary-free tiles; per-class pixels, tiles and
blocks per split; the Biodiversity gradient-step counts per cell; the OpenEarthMap and Biodiversity
class composition of the pre-training pool.

For each: your number, the repo's number, agree or differ. **A disagreement is the finding.**

**2. Hunt for functions that claim to compute the same thing and do not.**
This is where the worst defect was. `utils.spatial_blocks` and `build_spatial_split.support_blocks`
both partitioned ground into 950 m blocks and disagreed on Test B, because one scaled metres to
degrees using a mean latitude across two sites 50 km apart. That was fixed on 2026-07-26 — **verify
the fix, and then look for other pairs.** Candidates worth checking: boundary distance computed in
more than one place; foreground masking and ignore-index handling; per-class IoU in
`geoseg/utils/metric.py` against `scripts/analysis/bootstrap_metrics.metrics_from_cm`; tile-id
parsing and site inference; anywhere a pixel size or a CRS conversion appears.

**3. Check the tracked documents against the code, in both directions.**
Every falsifiable claim in `docs/METHODS_STATED_LIMITATIONS.md`, `docs/DECISIONS_REBUILD_2026-07.md`,
`docs/PRE_SUBMISSION_LEDGER.md` and `CLAUDE.md`. Both directions means: is what the document says true
of the code, AND does the code do something material the documents do not mention?

Pay particular attention to §6, §7, §9 and §10 of the methods file. §6 and §9 were corrected once
already after being wrong; §10 and §11 rest on quotations from three papers read in the last session,
and one of those was read from an arXiv version rather than the published proceedings.

**4. Try to make the campaign produce a wrong number and exit 0.**
Not crash — that is visible. Produce a plausible, wrong number, quietly. This is the failure mode that
matters, and the one this project has hit repeatedly.

Attack surfaces: the leakage gate; the manifest replay; the evaluation writer/reader path agreement;
the provenance checks that now guard metrics files; the figure pipeline; the softmax dump path; the
aggregation and its `--strict` flag; the two campaign launchers; the stage window logic in
`RUNBOOK.sh`. The withdrawn campaign's output has been moved to `_archive/withdrawn_campaign_2026-07-25/`
— **try to get any of it back into a reported result.**

**5. Review the last session's diff as a stranger's code.**

    git diff df8177d..HEAD          # ~70 files, ~4,700 insertions

That session fixed 42 findings and introduced at least two of its own: a provenance block reading an
argument that did not exist, which would have crashed all 40 runs, and a stage reorder that changed an
array but not the file, so the gate still ran after three cells had trained. Both were caught by
review, not by the author. Assume the same rate persists in what review did not reach.

Look especially at: the new `--self-test` blocks (do they test the thing, or the construction?); the
provenance checks added to readers; the block-grouping change in `utils.spatial_blocks` and everything
downstream of it; the stage C5 addition and the RUNBOOK reordering; and any comment written in that
session that asserts a number.
</the_five_dimensions>

<out_of_scope>
Do not re-report these. They are settled, recorded with evidence, and re-litigating them costs more
than it buys. If you believe one is wrong, write the argument in your report and change nothing.

- The transfer arm's 2.00x Biodiversity gradient-step confound and its second validation-selection
  pass. Declared before results exist (D12), stated in METHODS §1 and §2.
- No pre-registered threshold on rho, no dead band, no weak band (D18). rho is descriptive.
- 950 m being ireland2's composition range rather than the inland site's (D14, METHODS §4).
- The split clearing its adequacy floors at 5 of 10 grid phases (METHODS §5).
- Initialisation, seeds and bitwise reproducibility (D19, METHODS §9).
- `README.md`, `RUNBOOK.md`, `docs/DESIGN_NOTES.md` describing the old split — stale by decision,
  rewritten after the campaign.
- `manuscript/` — being rewritten after results exist. Report anything you find, change nothing.
- Convergence at 1,072 training tiles — assumed deliberately, to be read off the first run's curve.

Already found false and corrected; report only if the correction is itself wrong:
"all 432 parameter tensors identical across seeds"; D2's "18% of tiles lost to buffers"; D14's "950 m
on the full pool"; the boundary-free exclusion costing blocks.
</out_of_scope>

<constraints>
- Do NOT train. Do NOT run `run_campaign.sh` without `DRYRUN=1`. Do NOT touch the cluster or `sbatch`.
- Do NOT commit, checkout, stash or reset. Do NOT modify any tracked file — copy to scratch instead.
- Do NOT modify `data/`, `model_weights/`, `artifacts/` or `pretrain_weights/`. Use `cp -as` symlink
  trees for mutated splits.
- `evaluation/evaluation_results/` is gitignored scratch: you may create dummy files there and must
  delete exactly what you create.
- End with `git status --porcelain` and report it. It must show nothing you caused.
- Classify every finding: LAUNCH-BLOCKING (40 runs wasted or wrong numbers) / BEFORE-SUBMISSION /
  COSMETIC. A cosmetic finding reported as blocking costs a deadline.
</constraints>

<output_format>
1. **Can the campaign launch?** Yes or no, and if no, the shortest list of what must change.
2. **Dimension 1 table:** quantity | my derivation | repo's value | agree/differ. Every row.
3. **Disagreeing function pairs** found, with the input that separates them.
4. **False or missing statements** in the tracked documents.
5. **Attacks that produced a wrong number at exit 0** — and the attacks that failed, which matter
   just as much.
6. **Defects in the last session's diff.**
7. **Invented thresholds** you found anywhere, including in that session's work.
8. **What you could not verify**, stated plainly rather than inferred.
9. **Your confidence, and what would raise it.**
</output_format>
