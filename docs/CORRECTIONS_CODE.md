# CORRECTIONS — code and infrastructure

Companion to `docs/CORRECTIONS.md`, which is about the **manuscript**. This file is about the **repo**:
code, configuration, staging and the runbook. Nothing here changes a reported number.

Each entry records what was wrong, how it was found, and whether it is fixed. Anything still open is
marked so.

---

## FIXED — `.gitignore` matched directories only, so staging made every seed tree look dirty

**Found 2026-07-27**, while staging the campaign to Sonic. The launcher refused to submit:

    ERROR: the working tree is dirty. Commit or stash before launching:
        ?? data
        ?? pretrain_weights

**Cause.** The patterns were `data/` and `pretrain_weights/`. A trailing slash matches **directories
only**. On the laptop both are real directories, so the tree is clean and the bug is invisible. Step 2c
of `docs/audit/BRIEF_STAGE_TO_SONIC.md` creates both as **symlinks** into scratch, and git treats a
symlink as a file, so neither pattern matched. All ten seed trees read as dirty and the campaign could
not be launched.

**Fix.** Dropped the trailing slashes, with a comment saying why. `git check-ignore -v` confirms both
forms are still ignored when they are real directories, so nothing changes on the laptop.

**Why this class of bug keeps happening here:** the check and the thing it checks were written in the
same frame. Nobody ran `.gitignore` against a symlink because on the machine it was written on, one
never existed.

## FIXED — the staging brief described the OEM data chain as two levels; it is four

**Found 2026-07-27**, by running `find -xtype l` after following the brief exactly. It returned
**2,118 dangling symlinks** — 2,118 of the 3,190 tiles in the stage-2a pre-training pool.

**Cause.** `BRIEF_STAGE_TO_SONIC.md` said `data/openearthmap_relabelled` holds *"REAL FILES"* and that
`data/openearthmap_raw` must not be sent. Neither is true. Only the relabelled **masks** are real
(stage A8 wrote them). Every OEM **image** resolves down four levels:

    oem_combined_f1 -> openearthmap_relabelled -> openearthmap_filtered -> openearthmap_raw

The June 2026 staging never hit this because it ran `rsync -L`, which dereferenced everything and
copied the pixels — at the cost of 15 GB of duplication and 6,000 dangling links. When dereferencing
was banned and scratch was pruned on 2026-07-27, the only copy of those pixels went with it.

**Fix.** The brief now lists all four levels, states the real upload as ~6.6 GB rather than 98 MB,
carries the script that resolves the needed 2,118 files, and no longer forbids `openearthmap_raw`.

**Expected, not a defect:** `openearthmap_filtered/masks` shows 2,118 dangling links after staging.
Those are the original 9-class OEM masks, which nothing in B4..C5 reads, because `oem_combined_f1`'s
mask links point at the relabelled masks instead. The check that must be zero is
`find split_f1 oem_combined_f1 -xtype l`.

## FIXED — Test A's area did not reproduce

`utils.py:201`, `boundary_rate_ratio.py:48` and `METHODS_STATED_LIMITATIONS.md:178` gave Test A as
6.783 km² / 7.52 cells. The exact union of the 294 test footprints is **6.767 km²**, so the cell
equivalent is **7.50**. Corrected 2026-07-27. The 0.24% gap is unexplained; it is not the
degrees-to-metres constant, because the inland site is in projected metres.

## OPEN — the interaction term is labelled with a mechanism the design cannot isolate

`scripts/analysis/aggregate_seeds.py:87` reads `"transfer x sampler (interaction)"`. Factor A is the
pre-train-then-finetune **procedure**, not transfer — the transfer arm receives exactly 2.00x the
Biodiversity gradient steps. `docs/CORRECTIONS.md` Correction 1 fixed this naming for the main effect
and left the interaction untouched. Should read *procedure x sampler*.

Not urgent: it is a display label, it changes no number, and it is not in the campaign path.

## NOTED — validation is aggregated alongside the two test splits

`aggregate_seeds.py` `SPLIT_DIRS` includes `val`. That is fine as a convergence check, but validation
sits 256 m from training and is the split every checkpoint is selected on, so no headline number may
be drawn from it. Report it; never lead on it.

## NOTED — pyflakes is not installed in the project environment

The 2026-07-27 audit reported a clean `pyflakes` run over the campaign path. It could not be repeated
on 2026-07-27 during staging: the module is absent from both the `label-quality-ceiling` and `S2S_AI`
environments. `python -m py_compile` passes on every changed file, which catches syntax errors but not
undefined names. Add `pyflakes` to `environment.yaml` if that check is meant to be repeatable.
