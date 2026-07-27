# Brief — review the stage-E figure-pipeline audit of 2026-07-27

Paste everything below the rule into a **fresh chat**. It reviews a code audit that was run on
2026-07-27, one of whose findings has already been acted on. Your job is to check the findings, check
the change, and rule on the one open decision.

---

<role>
You are auditing an audit. Another agent ran a code audit of this repository on 2026-07-27, aimed at
two failure modes only: (a) the campaign crashes or trains the wrong thing, and (b) an analysis or
figure defect severe enough that the paper would have to be withdrawn after publication. It reported
a clean bill on the training path and one systemic defect in the figure path. It then changed one
file and moved 29 others.

You are not being asked to be polite about it. The prior agent got at least two things wrong and
corrected them mid-audit; assume there are more. It also worked while a second agent was committing
to the same tree, so some of what it read has since changed — check the current files, not its
quotations.

The repository's own recurring lesson applies to this review: a claim that has not been observed to
fail does not exist. Where a finding asserts something breaks, make it break, or say you could not.
</role>

<documents_to_read_first>

1. **`CLAUDE.md`**, sections "STATE AS OF 2026-07-26" and "Gate discipline" — the campaign state and
   the failure shape this repo keeps producing.
2. **`RUNBOOK.sh`**, stages C5, D and E — the pipeline the findings are about.
3. **`scripts/figures/build_all_figures.py`** — how stage E invokes each figure script. This is
   load-bearing for every finding below: it matters enormously whether it passes arguments.
4. **`docs/DO_NOT_ADD.md`** and **`docs/CORRECTIONS.md`** — what has already been settled, so you do
   not re-raise it.

Do not read the prior agent's other output (`docs/audit/METHODOLOGY_REVIEW_2026-07-27.md`) before
forming your own view of the code. It is a design review, not a code review, and it will bias you.

</documents_to_read_first>

<what_the_audit_verified_as_PASSING>

Listed so you do not spend the review re-running it. **Spot-check two or three; do not take the list
on trust, but do not redo all of it either.** Each was executed, not read.

| check | result |
|---|---|
| B4 sampler build | Runs; committed `artifacts/sampler_weights_clsbal_f1.tsv` is **bit-reproducible** (rebuilt to a temp path and diffed). Realised uplift 1.270× settlement / 2.844× semi-natural. |
| B4b full leakage preflight | Passes, 13 checks. Separation 256 / 768 / 1664 m; Test B 69 km from any train tile. |
| `pyflakes` over the whole campaign + analysis path | **No undefined names.** The D17 failure mode (a name referenced after its constant was deleted) is not live. |
| All 5 training configs, loaded in **separate processes** | `max_epoch=45`, `T_0=15/T_mult=2`, every path `_f1`-tagged, val n=173 for all five, steps/epoch 1595 vs 536 (the 2.00× in METHODS §1). Both transfer cells warm-start from **stage 2a, not 2b**. |
| Checkpoint filename contract | `ModelCheckpoint(dirpath=weights_path, filename=weights_name)` produces exactly the path `pretrained_ckpt_path` and B3's `require_file` demand. |
| Silent transfer failure | Guarded — `_load_student_weights_from_pl_ckpt` raises below 90% param match. |
| Lightning `-v1` filename collision | Cannot occur: `--force` unlinks **both** best and last, and the campaign passes `--force` by default. |
| Softmax dump layout | `seed_dir` resolves both layouts; `list_val_tiles` raises on an empty intersection. Test A and Test B share a dump directory but tile ids are disjoint. |
| `assert_metrics_provenance` | Five bad inputs constructed and **all rejected** (withdrawn campaign, val-passed-as-test, wrong cell, TTA on, Test B labelled Test A); valid control accepted. |
| `boundary_rate_ratio.py --self-test` | Passes, including the null control added on 2026-07-27. |
| Class-name vocabularies | Two exist (`compute_metrics.CLASS_NAMES_6` long, `taxonomy.STUDENT_CLASSES` short) but each is internally consistent and in the same index order; `load_metrics_file` indexes directly, so a mismatch would be a loud KeyError. |

**The most useful thing you can do with this table is disprove one of its rows.**

</what_the_audit_verified_as_PASSING>

<expected_failure_do_not_chase>

**One error will look like a bug and is not. Read this before you run any figure script.**

Commit `362f306` renamed the 1.5 m ratio so it could not be confused with rho.
`boundary_trimap_iou.py` now emits `contact_zone_vs_interior` / `contact_max_m`; it previously emitted
`boundary_vs_interior` / `boundary_max_m`. Any artefact from the withdrawn campaign still carries the
old keys, so pointing a figure script at withdrawn-campaign JSON raises:

    KeyError: 'contact_max_m'

**That is correct behaviour, not a defect.** It means the input predates the rename, i.e. it is
withdrawn data. Do not "fix" it by adding a fallback to the old key — a fallback would let withdrawn
numbers into a manuscript figure silently, which is the exact failure this repository has spent three
reviews closing. Regenerate the input from the current code instead, or accept that the figure cannot
be built until the campaign lands.

</expected_failure_do_not_chase>

<the_findings_to_review>

## F1 — Every campaign-dependent figure pointed at pre-rebuild locations — TWO OF FOUR NOW FIXED, REVIEW THE CHANGES

**Claim.** Stage C5 writes softmax to `analysis/seed_softmax`. The figure scripts default to
`sonic/results`, and to the **validation** split, which stage D's own comment excludes: *"Computed on
HELD-OUT ground only: validation is the split every checkpoint is selected on, so evidence drawn from
it is optimistic."*

| script | was | now | CLI override |
|---|---|---|---|
| `scripts/figures/uncertainty_quality.py` | `sonic/results` + `val/masks` | **FIXED** → `analysis/seed_softmax` + `test` | `--split`, `--softmax-root` added |
| `scripts/figures/reliability_ece.py` | `sonic/results` + `val/masks` | **FIXED** → `analysis/seed_softmax` + `test` | `--split`, `--softmax-root` added |
| `scripts/analysis/confident_learning_overlay.py:156-157` | `sonic/results` + `val/masks` | **UNCHANGED** — blocked by F3 | CLI default |
| `scripts/analysis/draft_boundary_overlay.py:290-291` | `sonic/results` + `val/masks` | **UNCHANGED** — blocked by F3 | CLI default |

**Why it mattered:** `build_all_figures.py` invokes all four **bare** —
`subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT)` — so the defaults are what stage E
runs. They were not merely a convenience for manual calls.

**Verify the claim:** confirm `build_all_figures.py` passes no arguments to these scripts. If it does
pass any, F1's premise is wrong and the two changes should be reconsidered. Then confirm nothing else
populates `sonic/results` during a campaign run.

**Review the two changes:** both now take `--split {test,external_test}` (validation rejected by
argparse) and `--softmax-root` defaulting to what stage C5 writes; both raise rather than retry a
missing input under mathtext. Both were observed to fail loudly, naming the producing stage. **Rule
on whether `test` is the right default given `main.tex` §sec:uncertainty currently describes all of
this on validation** — the prior agent asserts the repo already settled this (stage D: held-out
only), and that assertion is the load-bearing one.

**The two overlay scripts were deliberately not touched.** See F3.

## F2 — `analysis/label_ceiling/stats_<cell>.json` had no producer — NOW ADDED TO STAGE D, REVIEW IT

**The defect.** `uncertainty_quality.py` reads it; it is written by
`scripts/analysis/seed_disagreement.py:377`; `grep -n seed_disagreement RUNBOOK.sh` returned nothing,
so stage E could never build that figure.

**Note a correction the prior agent made:** it first attributed this file to `figure_label_ceiling.py`.
That was wrong. `seed_disagreement.py` is the producer. The conclusion survived the correction, but
treat the finding as one that was already stated carelessly once.

**The change.** Stage D now runs `seed_disagreement.py` per held-out split, beside the trimap curve:

    --softmax-root analysis/seed_softmax  --mask-dir "$SPLIT_ROOT/$SPLIT/masks"
    --cell stage1_baseline --cell stage3_clsbal  --out-dir "analysis/label_ceiling/$SPLIT"

`bash -n RUNBOOK.sh` passes and the script accepts those arguments. **Review whether both cells are
the right set** — `reliability_ece.py` needs `stage1_baseline` and `stage3_clsbal`; the uncertainty
figure needs only `stage3_clsbal`. And whether stage D should emit this for `external_test` at all,
given nothing currently consumes the Test B copy.

## F3 — A hand-picked narrative tile is now a TRAINING tile. SETTLED — do not re-verify

Both overlay scripts hardcode `DEFAULT_TILES = ["biodiversity_1969", "biodiversity_2126"]`, chosen
for narrative reasons written into the manuscript captions — *"Top: an accurate tile; bottom: a hard
semi-natural-grassland tile."* On the rebuilt split, confirmed against `data/split_f1/*/images/`:

    biodiversity_1969  -> test      (fine)
    biodiversity_2126  -> train     <- bottom row of TWO manuscript figures
    biodiversity_1403  -> test      (fine; a --save-map-tiles default)

**The facts are established and were independently checked. Spend no review time re-deriving them.**

The consequence is what matters for F1: the obvious mechanical fix — flip the default mask dir from
`val` to `test` — makes 1969 resolve and 2126 not. So the two overlay figures cannot be repaired by
editing a path, which is why they were left at their old defaults while their two siblings were
fixed. Leaving them visibly broken is preferable to half-fixing them into a state where one panel
resolves and the other silently does not.

**The tile re-selection is not merely undesirable now — it is impossible now, by construction.** The
captions claim properties of *predictions*: `draft_boundary_overlay.py:69-71` documents 1969 as the
accurate tile at *"3.7% of fg pixels differ"* and 2126 as the hard tile at *"23% differ"*. No model
has been trained on this split, so no such percentage exists for any tile. You cannot choose "the
tile where 3.7% of foreground pixels differ" before there are predictions. **This is a post-campaign
task and belongs on the post-campaign checklist, not in a pre-stage fix.**

## F4 — `boundary_limited_error.py` read a path stage D never writes — ALREADY CHANGED, REVIEW THE CHANGE

**The defect.** Stage D writes `analysis/label_ceiling/$SPLIT/boundary_trimap_<cell>.json` for SPLIT
in {test, external_test}. The figure read `analysis/label_ceiling/boundary_trimap_<cell>.json`, with
no argument that could supply a split. Nothing in the shipped pipeline writes that path — only a bare
`boundary_trimap_iou.py` run taking its own default `--out-dir`, which is Test A only and records no
split.

**The change made on 2026-07-27** (`git diff scripts/figures/boundary_limited_error.py`, +38/−4):
- new `trimap_json(root, split, cell)` resolving the per-split path;
- new `--split {test,external_test}` argument, default `test`;
- a `SystemExit` naming RUNBOOK stage D when the file is absent, and refusing the un-split legacy
  path explicitly rather than plotting it;
- `main()` no longer swallows a `SystemExit` into the usetex retry, which previously would have
  reported "usetex failed" for a missing file.

It was observed to fail on a missing input, observed to refuse a planted legacy file, and observed to
reject `--split val`.

**Review it as a code change, not as a finding.** The default value is close to irrelevant —
`build_all_figures.py` invokes the script **bare**, so no default it carries will ever be overridden
in the shipped pipeline. The real question is therefore about the pipeline, not the argument:

- **Should stage E build this figure for both held-out splits?** Stage D already computes the trimap
  curve for `test` and `external_test`. If the paper reports only Test A, stage D is doing work
  nothing consumes; if it should report both, `build_all_figures.py` needs to invoke the script twice
  with distinct output names, and `docs/FIGURES.md` and `main.tex` need a second figure. That is a
  RUNBOOK/`build_all_figures.py` change, and neither has been made.
- Is the refusal of the un-split legacy path too strict for a legitimate local debug run?
- Does `docs/FIGURES.md` need updating either way?

</the_findings_to_review>

<the_change_to_the_working_tree>

Two things were changed. Both are reversible; neither touches training, analysis numbers, or the
manuscript source.

1. **`scripts/figures/boundary_limited_error.py`** — F4 above. The only code change.
2. **29 figure files moved to `_archive/stale_figures_pre_campaign/`** (not deleted) from
   `figures/` and `manuscript/Figures/`: the 8 campaign-dependent PDFs built from the withdrawn
   leaking campaign, plus 4 unreferenced orphans (`boundary_distance`, `ceiling_band`,
   `factorial_design`, `rgb_tiles`) and their `.png` siblings.

   **This is stated so you know the state of the tree, not for you to rule on.** The author has
   decided it. Consequence, for your awareness only: `main.tex` now references 8 figures that do not
   exist, so `latexmk` fails on a missing graphic instead of compiling silently against a withdrawn
   figure. Restoring from `_archive/stale_figures_pre_campaign/` reverses it.

</the_change_to_the_working_tree>

<the_open_decision>

F1's fix is mechanical for `uncertainty_quality.py` and `reliability_ece.py` — point them at
`analysis/seed_softmax` and a held-out split, and add the CLI overrides they lack.

It is **not** mechanical for the two overlay scripts, because of F3: the illustrative tiles must be
re-chosen from the new held-out split, and the choice is tied to caption text describing what each
tile shows. Rule on:

1. Which split each of the four figures should be computed on. The prior agent asserts the repo has
   already decided this (held-out only, per stage D) and that it is an unapplied decision rather than
   an open one. Test that claim — the calibration and uncertainty figures may have a different
   rationale from the boundary figure, and §sec:uncertainty in `main.tex` currently describes all of
   this on validation.
2. Whether new narrative tiles are chosen now or after the campaign lands, given the captions depend
   on properties (error fraction, class content) that cannot be known until there are predictions.

</the_open_decision>

<instructions>

1. **Quote before you judge.** Reproduce the line you are ruling on, from the current file, with its
   line number. Two files in this brief were edited by a concurrent agent during the audit; some
   quotations may already be stale.
2. **Rule on F1, F2 and F4** as CONFIRMED, OVERSTATED, or WRONG, with the evidence. F3's facts are
   settled and independently checked — do not spend review time on them.
3. **Try to break the passing table.** Pick the two rows you find least plausible and attack them.
4. **Say what the prior agent missed.** It looked at the training path, the metrics path and the
   figure path. It did **not** examine: `evaluation/aggregate_metrics.py`, `export_final_test_table.py`,
   `compute_metrics.py` beyond its mIoU definition, `graphical_abstract_panels.py`, or any of the
   `a1`–`a6` supplementary analyses. Any of those could hold the same defect class.
5. **Do not re-litigate the design.** Whether the study can support its claims is a separate review
   (`docs/audit/METHODOLOGY_REVIEW_2026-07-27.md`). This one is about whether the code does what the
   pipeline says it does.

</instructions>

<constraints>
- **REPORT ONLY. Change nothing.** No edits to any file, no commits, no restores from `_archive/`.
  This holds even where a fix looks mechanical — F1's repair for two of the four scripts is a
  two-line change and you must still only describe it. A second agent is editing this tree
  concurrently and a third is running a design interrogation; an edit from you would collide.
- The campaign has not been staged or launched. Nothing has trained on the current split.
- If you disagree with the change already made in F4, say so and propose the alternative. Do not
  revert it.
- Do not chase `KeyError: 'contact_max_m'` — see the expected-failure note above.
</constraints>

<output_format>

```
## Verdict on each finding
F1 / F2 / F4 — CONFIRMED | OVERSTATED | WRONG, one paragraph each, with the quoted line.
(F3 is settled; do not rule on it.)

## The passing table
Which rows you attacked and what happened.

## The change in F4
Is scripts/figures/boundary_limited_error.py right, and should stage E build both held-out
splits rather than one? Describe the RUNBOOK / build_all_figures.py change; do not make it.

## The open decision
Your ruling on the split per figure, and on when narrative tiles get re-chosen.

## What the prior audit missed
The highest-value thing it did not look at. If there is nothing, say so.
```

</output_format>
