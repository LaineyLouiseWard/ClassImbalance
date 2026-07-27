# Brief — stage the campaign to Sonic and launch it

Paste everything below the rule into a **fresh chat**. Run it only after
`docs/audit/BRIEF_FINAL_AUDIT.md` has come back and anything launch-blocking is fixed.

**Revised 2026-07-27.** Three things in the previous version were wrong or missing and all three
change the plan: the runtime figure, the concurrency assumption, and the state of scratch. Every
number below was measured on the cluster on 2026-07-27, not estimated. See
`## What changed on 2026-07-27` at the end for the diff.

---

<role>
You are staging a 40-run GPU campaign onto a shared HPC cluster and launching it. The code is done and
audited; your job is to put the right bytes in the right places, prove they are right, and submit.

The expensive failure here is not a crash. It is ten array tasks that queue, wait for an L40S, start,
and die in one second on a missing directory — or worse, run for twenty hours against subtly wrong
data. So: **verify each stage before starting the next**, and prefer a check that takes a minute over
an assumption that costs a queue slot.

Do not modify the repository. If something is wrong with the code, stop and report it rather than
patching it here — the commit that runs must be the commit that was audited.
</role>

<the_runtime_reality>

**Read this before planning anything around a deadline. The previous version of this brief said each
run takes ~20 h; that was the Slurm `--time` ceiling, not a runtime, and it implied 40 × 20 h.**

**You are capped at four concurrent GPUs.** Measured 2026-07-27:

    QOS normal:  MaxTRESPU = cpu=200, gres/gpu=4

`MaxTRESPU` is *max TRES per user*. The association is `shared_acc | normal`. So `--array=0-9` runs
**four tasks at a time** regardless of how idle the cluster is, and the other six queue. There are 16
L40S GPUs on the cluster (8 nodes × 2); none of that matters, the cap binds first.

Confirmed empirically from the previous 10-seed campaign, job `454217` on 2026-06-27:

| | |
|---|---|
| four tasks start together | 10:41:27 |
| task 5 starts | 15:14:37 — **exactly** when task 3 ended |
| task 6 starts | 16:12:16 — exactly when task 1 ended |
| task 8 starts | 19:20:42 — exactly when task 4 ended |
| per-task elapsed | **4 h 25 min – 5 h 32 min** |
| **10 tasks, submit to last finish** | **13 h 55 min** |

Strict backfill on a four-slot cap. So the wall clock is **three waves**, and everything turns on the
per-task time:

- at ~5.5 h/task (the 2026-06 rate) → **~17 h total**
- at 14 h/task (the pessimistic estimate) → **~42 h total**

The 2026-06 rate is the better guide. Training work per seed is now roughly 7 baseline-equivalents on
1,072 tiles against roughly 5 on 1,706 last time — about **0.88× the old load**. Stage 2a trains
**once** per seed and both transfer cells warm-start from it (`RUNBOOK.sh` B2 → B3, B5), so there is
no duplication to remove. The genuinely new cost is the 4-cell × 2-split softmax dump, which is I/O
heavy but not eight hours.

**What to do with that:** submit the full `--array=0-9` rather than a pilot seed. The cap serialises
you anyway, so holding nine back buys nothing and costs queue position, and you get real per-task
timings from the first wave within about six hours — with `scancel` available on the pending tasks if
they come in slow.

**Watch the 20 h `--time`.** Comfortable at 12–16 h, no margin above that. A task that exceeds it is
killed mid-collect.

**Do not wait on anything to raise the cap.** A `boost` QOS exists at `gres/gpu=8` and the account is
not associated with it, but at ~5.5 h/task three waves is ~17 h and that is fine — boost would save
about six hours on a job that already fits. **This is not a prerequisite. Launch without it.** There
is also no workaround worth trying: the cap is on total GPUs, not GPU type, so dropping
`--constraint=l40s` buys no extra slot and only scatters the seeds across different hardware.

If the campaign does turn out to be slow, the lever is cutting seeds at the step-5 decision point,
not chasing more GPUs.

</the_runtime_reality>

<state_of_scratch>

**Verified on the cluster 2026-07-27. Check it again before you rely on it — scratch is purgeable and
this section is a snapshot, not a guarantee.**

`/home/people/<user>/scratch/lqc` is the campaign root (`SONIC_SCRATCH`). It currently holds **17 GB**,
already staged and verified:

| path | size | state |
|---|---|---|
| `lqc/env` | 7.9 GB | **verified working**: `torch 2.9.1+cu128`, `pytorch-lightning 2.3.0`, `python 3.11.14` — exactly the pins in `environment.yaml`. `conda activate` resolves correctly from this prefix. |
| `lqc/data_stage/data/biodiversity_split` | 8.4 GB | **verified byte-identical to the laptop**: `train 1706 / val 219 / test 218`, 4,286 files, first three image md5s match (`ad512bb3bfac`, `2a13c98bfdfe`, `635113ed0198`). |
| `lqc/data_stage/pretrain_weights` | 542 MB | `stseg_base.pth` (464 MiB) + the teacher `.pth`. |
| `lqc/logs` | empty | ready for the array's `--output`. |

This came from renaming the June campaign root `classimb` → `lqc` and pruning 403 GB of withdrawn
artefacts on 2026-07-27. The withdrawn campaign's result JSONs were archived to the laptop at
`_archive/withdrawn_campaign_sonic_results/` (116 files, 2 MB) before the delete, because
`rgbnir_results/` backs the near-infrared claim in the Discussion.

**So step 1 is a 98 MB top-up, not a 9 GB upload, and step 2d is skipped entirely.** If any of the
above is missing when you check, fall back to `## Appendix — staging from nothing`, which is complete
and assumes an empty scratch.

**Note on the `train/val/test` subdirectory names in `biodiversity_split`.** They are the *old* random
split's directory layout and that is correct — the pool's directory structure is what the `split_f1`
symlinks resolve through, even though the old assignment is discarded.
`docs/METHODS_STATED_LIMITATIONS.md` §8 records why. Do not "fix" it.

**Disk is not a constraint.** `/scratch` is BeeGFS, 559 TB with 61 TB free, and BeeGFS quota is an
unlicensed Enterprise feature here so no per-user quota is enforced. The per-seed checkpoint collect
in `campaign.slurm` checks `df` for 2× the checkpoint size (~120 GB across all ten seeds) before its
first copy; against 61 TB free that check passes trivially. It exists so a half-copied checkpoint
never looks like a successful collection.

</state_of_scratch>

<what_you_are_staging>
`label-quality-ceiling`. Four factorial cells × ten seeds (42–51) = 40 training runs on a fixed
FT-UNetFormer, testing two data-curation interventions. The paper's contribution is a boundary-label
diagnosis, not the interventions.

Each **array task** is one seed and runs `RUNBOOK.sh --from B4 --to C5`: the sampler build, the full
leakage gate, five training configs (baseline, stage 2a pre-train, stage 2b finetune, sampler-only,
stage 3 clsbal), evaluation on validation / Test A / Test B, and the per-seed softmax dumps the
boundary evidence is computed from. Then it collects metrics, softmax dumps and checkpoints to
`$SONIC_SCRATCH/results/`.

**The launcher is `sonic/campaign/submit_campaign.sh`.** It is the only supported path. Do NOT use
anything else under `sonic/` — those are the pre-rebuild scripts, they hard-code one person's student
number, and `sonic/10_submit_final_campaign.slurm` bypasses the preflight gate entirely and tests for
artefacts that were withdrawn. `sonic/*` is gitignored EXCEPT `sonic/campaign/`, which is the tracked
path and the one you want.

**What the launcher refuses to do**, so you know what it is checking rather than being surprised:
- it refuses on a dirty working tree, because the array pins all ten seeds to one commit
- it refuses if `$SONIC_SCRATCH/env` or any of `$SONIC_SCRATCH/seed42 … seed51` is missing
- each array task re-checks the commit it is actually running and aborts if it differs
- each array task runs the full leakage gate BEFORE training and aborts if it fails
- the collect step refuses on a missing metrics file, an empty softmax directory, a missing
  checkpoint, or insufficient disk — and it checks disk *before* the first copy

**Environment variables it reads** (all optional except where noted):

    SONIC_USER        cluster username                          (default: $USER)
    SONIC_SCRATCH     campaign root on scratch                  (default: /home/people/$SONIC_USER/scratch/lqc)
    SONIC_ACCOUNT     Slurm account                             (default: shared_acc)
    SONIC_MAIL        address for END,FAIL mail                 (optional; no mail if unset)
    SPLIT_TAG         which split to run                        (default: f1)
    BATCH_VARIANT     b2 = batch 2 / lr 3e-4 (default), b4 = batch 4 / lr 6e-4
    COLLECT_PRETRAIN  1 = also collect the stage-2a checkpoint  (default: 0)

`DRYRUN` is anything other than empty or `0` — note that `DRYRUN=false` is a dry run, not a
submission.
</what_you_are_staging>

<the_data_layout_that_matters>
**Read this before you rsync anything. Getting it wrong is the single most likely way to waste a day,
and it has already happened once — see the warning at the end of this section.**

The campaign's data is a two-level chain of RELATIVE symlinks, not three independent copies:

    data/biodiversity_split/       8.4 GB   REAL FILES — the 2,143-tile pool      [already on scratch]
    data/openearthmap_relabelled/   69 MB   masks are REAL FILES; images are SYMLINKS
                                            -> ../../openearthmap_filtered/images/...             [UPLOAD]
    data/openearthmap_filtered/     17 MB   SYMLINKS -> ../../openearthmap_raw/OpenEarthMap/...   [UPLOAD]
    data/openearthmap_raw/          25 GB   REAL FILES — only the 2,118 filtered images
                                            are needed, 6.5 GB                                    [UPLOAD SUBSET]
    data/split_f1/                  12 MB   SYMLINKS -> ../../../biodiversity_split/<split>/...   [UPLOAD]
    data/oem_combined_f1/           17 MB   SYMLINKS -> ../../../split_f1/...  and
                                                     -> ../../../openearthmap_relabelled/...      [UPLOAD]
    pretrain_weights/              542 MB   REAL FILES — stseg_base.pth + teacher [already on scratch]

So `data/oem_combined_f1` points at `data/split_f1`, which points at `data/biodiversity_split`.

**The OEM side is FOUR levels deep, not two, and the previous version of this brief got it wrong.**
Corrected 2026-07-27 after it produced 2,118 dangling symlinks on a real staging run:

    oem_combined_f1 -> openearthmap_relabelled -> openearthmap_filtered -> openearthmap_raw

Only the relabelled **masks** are real files (stage A8 wrote them). Every OEM **image** resolves all
the way down to `openearthmap_raw`. Send the resolved subset — 2,118 files, 6.5 GB — with paths
preserved, not the whole 25 GB and not dereferenced. Build the list by resolving every symlink under
`data/oem_combined_f1` and keeping the targets under `openearthmap_raw`. On a home connection at
~650 kB/s this is about **three hours**, so start it before anything else.

`openearthmap_filtered/masks` will still show 2,118 dangling links afterwards. That is correct and
expected: they point at the original 9-class OEM masks, which nothing in B4..C5 reads, because
`oem_combined_f1`'s mask links go to the relabelled masks instead. The check that must be zero is
`find split_f1 oem_combined_f1 -xtype l`.

Four consequences:

1. **You are uploading ~6.6 GB, not 9 GB and not 98 MB** — `openearthmap_relabelled`, `split_f1`,
   `oem_combined_f1` and `openearthmap_filtered` (~98 MB together), **plus the 6.5 GB of resolved
   OpenEarthMap images** the chain bottoms out in. The 8.4 GB Biodiversity pool and the 542 MB of
   weights are already there and verified.
2. **Preserve the symlinks. Do NOT use `rsync -L`.** Dereferencing turns 28 MB of links into ~13 GB of
   duplicated pixels and breaks the guarantee that the pre-training pool and the training split are
   literally the same bytes — a property the leakage gate checks.
3. **The relative paths only resolve if all four directories sit under one common `data/` parent.**
   They must land in `$SONIC_SCRATCH/data_stage/data/`, beside the existing `biodiversity_split`.
4. **`notes/` must never travel.** It is gitignored, so a `git clone` will not carry it — which is one
   reason the code arrives by clone rather than by rsync. If you rsync the repo for any reason,
   exclude it explicitly.

> **This actually happened.** The June staging was done with dereferencing on. It left a 15 GB
> `biodiversity_oem_combined` of duplicated pixels and **6,000 dangling symlinks** in `data_stage`.
> Both were deleted on 2026-07-27. The `find -xtype l | wc -l` check in step 1 exists because of it —
> run it, and if it is non-zero, fix the layout rather than dereferencing again.

`data/biodiversity_raw` (9 GB) is **not** needed: it is an A-stage input and the A stages have already
run. `data/dem` (76 MB) is used only by a Discussion-side analysis, not by B4..C5. **`openearthmap_raw`
IS needed** — see above; send the 2,118-file, 6.5 GB subset, not the whole 25 GB.

**Every artefact the campaign needs is committed** and arrives with the clone: the split manifest, the
sampler weights, the augmentation list, the teacher confusion, the boundary denominators, the
correlogram, and — added 2026-07-27 — `normalisation_stats_<tag>.json`. You do not need to transfer
anything from `artifacts/`.

> **Check this holds before every launch, don't assume it.** The normalisation stats were briefly
> untracked, and because the image reader hard-fails without them and each seed 43-51 is a
> `git worktree add` (tracked files only), all forty runs would have died at their first batch after
> queueing. `campaign.slurm` now asserts the file up front, and RUNBOOK stage B4 builds it if absent.
</the_data_layout_that_matters>

<procedure>

## 0. Before you touch the cluster

    export PATH="$HOME/miniconda3/envs/label-quality-ceiling/bin:$PATH"
    cd /home/lainey/Documents/Github/label-quality-ceiling
    git status --porcelain          # MUST be empty
    git rev-parse HEAD              # write this down; it is the campaign commit
    git log --oneline -1

Confirm the audit's blocking findings are fixed and committed. Run the gate locally one last time:

    SPLIT_TAG=f1 AUG_LIST=artifacts/train_augmentation_list_f1.json \
      TEACHER_CONFUSION_NPZ=artifacts/teacher_oem_gt_confusion_f1.npz PYTHONPATH=. \
      python scripts/data_prep/assert_no_split_leakage.py --split-root data/split_f1 \
        --oem-root data/oem_combined_f1 --sampler-tsv artifacts/sampler_weights_clsbal_f1.tsv

**Push the campaign commit to GitHub**, because step 2 clones from there:

    git push origin main

## 1. Confirm what is already on scratch, then top up (~98 MB)

**First, check the snapshot in `<state_of_scratch>` still holds.** On Sonic:

    BASE=$HOME/scratch/lqc
    STAGE=$BASE/data_stage
    du -sh $BASE/env $STAGE/data/biodiversity_split $STAGE/pretrain_weights 2>/dev/null
    $BASE/env/bin/python -c "import torch, pytorch_lightning as pl; print(torch.__version__, pl.__version__)"
    for s in train val test; do printf "%-8s %s\n" $s $(ls $STAGE/data/biodiversity_split/$s/images | wc -l); done
    # expect  7.9G / 8.4G / 542M ;  2.9.1+cu128 2.3.0 ;  train 1706  val 219  test 218

**If any of that is missing or wrong, stop and use the Appendix instead.** Do not half-reuse.

Then, from the laptop:

    SONIC_USER=<username>            # ask the user; do not guess
    HOST=login.ucd.ie
    BASE=/home/people/$SONIC_USER/scratch/lqc
    STAGE=$BASE/data_stage

    ssh $SONIC_USER@$HOST "mkdir -p $STAGE/data $BASE/logs"

    # -a preserves symlinks as symlinks. NEVER add -L here.
    rsync -avh --partial --progress \
      data/openearthmap_relabelled data/openearthmap_filtered data/split_f1 data/oem_combined_f1 \
      "$SONIC_USER@$HOST:$STAGE/data/"

Then the 6.5 GB the OEM chain resolves to. Build the list first, so you send the 2,118 files the pool
actually needs rather than all 25 GB of `openearthmap_raw`:

    python - <<'EOF'
    import os, glob
    need = set()
    for f in glob.glob('data/oem_combined_f1/**/*', recursive=True):
        if os.path.islink(f):
            t = os.path.realpath(f)
            if '/openearthmap_raw/' in t and os.path.exists(t):
                need.add(os.path.relpath(t, os.getcwd()))
    open('/tmp/oem_raw.lst', 'w').write('\n'.join(sorted(need)) + '\n')
    print(len(need), 'files', round(sum(os.path.getsize(f) for f in need) / 1e9, 2), 'GB')
    EOF
    # expect  2118 files 6.5 GB
    rsync -ah --partial --info=progress2 --files-from=/tmp/oem_raw.lst . \
      "$SONIC_USER@$HOST:$STAGE/"

`rsync` is resumable — if it drops, re-run the same command. Start the 6.5 GB before step 2; the seed
trees clone from GitHub and do not need the data, so the two overlap.

**Verify before moving on.** On Sonic:

    cd $STAGE/data
    for s in train val test external_test; do printf "%-14s %s\n" $s $(ls split_f1/$s/images | wc -l); done
    # expect  train 1072   val 173   test 294   external_test 191
    ls oem_combined_f1/train/images | wc -l        # expect 3190
    find split_f1 oem_combined_f1 -xtype l | wc -l # expect 0  (no dangling symlinks)
    ls -lh $STAGE/pretrain_weights/stseg_base.pth  # expect ~464M

A non-zero dangling count means the symlinks did not resolve — almost always because the four data
directories are not under one common parent. Fix the layout, do not dereference.

The two sets of counts differ by design: `biodiversity_split` is the 2,143-tile pool under the old
directory names, and `split_f1` is the spatially blocked assignment over it (1072/173/294 plus 191
external).

## 2. Build the ten seed trees, on the LOGIN node

The login node has internet; compute nodes do not. Nothing here needs internet except the clone
(`HF_HUB_OFFLINE=1` is set at submit and every cell loads a local checkpoint or a local `.pth`).

    ssh $SONIC_USER@$HOST
    BASE=$HOME/scratch/lqc
    STAGE=$BASE/data_stage
    COMMIT=<the commit you wrote down in step 0>

    # 2a. base checkout = seed 42
    git clone https://github.com/LaineyLouiseWard/label-quality-ceiling.git $BASE/seed42
    git -C $BASE/seed42 checkout --detach $COMMIT

    # 2b. seeds 43-51 as detached worktrees at the SAME commit
    for S in 43 44 45 46 47 48 49 50 51; do
      git -C $BASE/seed42 worktree add --detach $BASE/seed$S $COMMIT
    done

    # 2c. one copy of the data and the weights, symlinked into every seed tree
    for S in 42 43 44 45 46 47 48 49 50 51; do
      ln -sfn $STAGE/data            $BASE/seed$S/data
      ln -sfn $STAGE/pretrain_weights $BASE/seed$S/pretrain_weights
    done

**2d. The conda env already exists at `$BASE/env` and is verified — skip the build.** It survived the
`classimb → lqc` rename: `bin/python` is a real binary rather than a shebang script, so the path
change does not break it, and `conda activate $BASE/env` was confirmed working on 2026-07-27. Build it
only if step 1's check failed; the Appendix has the command.

**Verify before moving on:**

    module load anaconda3/2025.12-1
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda run -p $BASE/env python -c "import torch, pytorch_lightning as pl; print(torch.__version__, pl.__version__)"
    for S in 42 43 44 45 46 47 48 49 50 51; do
      printf "seed%-3s %s %s\n" $S "$(git -C $BASE/seed$S rev-parse --short HEAD)" \
        "$(ls $BASE/seed$S/data/split_f1/train/images 2>/dev/null | wc -l)"
    done
    # every row must show the SAME short commit and 1072

The repo uses `pytorch_lightning`, not the unified `lightning` namespace — import that one or a
healthy env will look broken. (`import lightning` fails in this env, and that is expected.)

## 3. Prove a GPU node can actually run it, before spending 40 slots

    srun -p gpu --gres=gpu:1 --constraint=l40s -t 00:15:00 --pty bash
    module load anaconda3/2025.12-1
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate $BASE/env
    cd $BASE/seed42
    export SPLIT_TAG=f1 PYTHONPATH=. HF_HUB_OFFLINE=1
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    SPLIT_TAG=f1 AUG_LIST=artifacts/train_augmentation_list_f1.json \
      TEACHER_CONFUSION_NPZ=artifacts/teacher_oem_gt_confusion_f1.npz \
      python scripts/data_prep/assert_no_split_leakage.py --split-root data/split_f1 \
        --oem-root data/oem_combined_f1 --sampler-tsv artifacts/sampler_weights_clsbal_f1.tsv
    exit

If the gate fails here it will fail in all ten tasks. Fix the staging, not the gate. This `srun` may
itself wait for a slot — the four-GPU cap applies to interactive jobs too.

## 4. Dry run, then submit

From the laptop, or from `$BASE/seed42` on Sonic:

    SONIC_USER=<username> SONIC_SCRATCH=$BASE DRYRUN=1 bash sonic/campaign/submit_campaign.sh

Read the plan it prints: commit, scratch root, account, split tag, batch variant, 10 tasks, 4 cells,
40 runs. Confirm the commit matches step 0. **If it refuses, it is telling you something real** —
a dirty tree, or a missing seed directory. Fix that; do not work around it.

Then, for real:

    SONIC_USER=<username> SONIC_SCRATCH=$BASE bash sonic/campaign/submit_campaign.sh
    squeue -u $SONIC_USER

**Expect four RUNNING and six PENDING.** That is the QOS cap, not a failure. Do not resubmit the
pending tasks and do not resize the array to compensate.

## 5. Watch the first wave, and STOP

Within the first hour:

    tail -f $BASE/logs/lqc-campaign_seed0_*.out

Confirm, in order: the provenance line naming the commit; `LEAKAGE PREFLIGHT PASSED`; `[B4]` building
the sampler; `[B4b]` passing; then `[B1]` starting to train. The gate runs before any training by
design — if you see `[B1]` before `[B4b]`, stop and report it.

**When the first seed finishes, stop and check three things before the rest complete.** Under the
four-GPU cap this is a natural checkpoint: six tasks have not started yet, so a problem found here
costs four runs, not forty.

1. **Elapsed time.** `sacct -j <jobid> -X -o JobID,Elapsed,State`. This is the number the whole
   schedule depends on and it has never been measured for this configuration. At ~5 h you finish in
   about 17 h; at ~14 h you finish in about 42 h. If it is the latter, decide *now* whether to cancel
   the pending tasks and cut seeds.
2. **Convergence.** The 45-epoch budget was set for 1,706 training tiles and there are now 1,072. Read
   the validation curve from `$BASE/seed42/lightning_logs/`. If `val_mIoU` is still climbing at epoch
   45, the baseline is under-trained and every intervention will look better than it is. That is worth
   knowing before spending the other runs, and `CLAUDE.md` lists it as still open.
3. **Provenance.** `cat $BASE/seed42/model_weights/biodiversity/*/run_provenance_seed42.json` — commit,
   dirty flag, GPU, precision, torch and lightning versions. Confirm the commit is the campaign commit
   and `dirty` is `false`.

**While the array runs, the wall clock is free.** Two analysis fixes from
`docs/audit/METHODOLOGY_REVIEW_2026-07-27.md` are CPU-only, operate on the stage-C5 softmax dumps and
are on the critical path afterwards — write them now rather than after the array lands: per-seed
accumulation in `boundary_trimap_iou.py`'s A2 block, and a foreground-only boundary variant for
Test B. Neither changes what the campaign computes.

## 6. Fetch and aggregate

The array writes to `$BASE/results/`: `{val,test,external}/seed<N>_<cell>.json`,
`softmax/<cell>/seed<N>/`, and `checkpoints/<cell>/seed<N>.ckpt`.

    rsync -avh "$SONIC_USER@$HOST:$BASE/results/" ./sonic/results_f1/

    SPLIT_TAG=f1 PYTHONPATH=. python scripts/analysis/aggregate_seeds.py \
      --results-dir          sonic/results_f1/val \
      --test-results-dir     sonic/results_f1/test \
      --external-results-dir sonic/results_f1/external \
      --strict

`--strict` fails if any of 4 cells × 3 splits × 10 seeds is missing. The aggregator also checks each
file's recorded checkpoint against the cell and split tag being asked for, so a metrics file from
another campaign is rejected rather than averaged in.

**Pull the checkpoints too, or at minimum confirm they were collected.** They are ~61 GB and they are
what makes the campaign repeatable: re-evaluation on a corrected split, any new figure needing
predictions, and the step-matched control D12 names as its own reversal condition all branch from
them. Until 2026-07-27 they were not collected at all and would have died with the scratch purge.

</procedure>

<constraints>
- Do NOT modify the repository. The commit that runs must be the commit that was audited.
- Do NOT use any script under `sonic/` except `sonic/campaign/`.
- Do NOT use `rsync -L` on the data.
- Do NOT transfer `notes/` or `data/biodiversity_raw`. Send only the resolved 6.5 GB subset of
  `data/openearthmap_raw`, never the full 25 GB.
- Do NOT resubmit or resize the array to work around six PENDING tasks. That is the QOS cap.
- Do NOT delete anything under `$SONIC_SCRATCH` without inspecting it first. The 2026-07-27 prune kept
  17 GB out of 420 GB precisely because two of the survivors were expensive to recreate.
- Ask the user for the cluster username and the Slurm account rather than guessing. Do not put either
  into a tracked file.
- If a gate refuses, report it and stop. Every refusal in this pipeline was added because the thing it
  refuses actually happened.
</constraints>

<output_format>
1. **Where staging got to**, step by step, with the verification output for each.
2. **Anything that refused**, what it said, and what you did.
3. **The submitted job id**, and the plan the dry run printed.
4. **The first wave's first hour**: gate output, stage order, and whether training started.
5. **The first completed seed's elapsed time**, and the resulting projection for all ten.
6. **Anything you could not verify** from the laptop side.
</output_format>

---

## Appendix — staging from nothing

Use this if `$SONIC_SCRATCH` is empty, or if step 1's verification failed. It assumes nothing survives
on scratch and supersedes steps 1 and 2d above.

**Upload ~9 GB instead of 98 MB:**

    rsync -avh --partial --progress \
      data/biodiversity_split data/openearthmap_relabelled data/split_f1 data/oem_combined_f1 \
      "$SONIC_USER@$HOST:$STAGE/data/"
    rsync -avh --partial --progress pretrain_weights "$SONIC_USER@$HOST:$STAGE/"

Then verify exactly as in step 1, plus:

    for s in train val test; do printf "%-8s %s\n" $s $(ls $STAGE/data/biodiversity_split/$s/images | wc -l); done
    # expect  train 1706  val 219  test 218   (the POOL's old directory names — correct, see above)

**Build the conda env** (login node only; it needs internet):

    module load anaconda3/2025.12-1
    conda tos accept --override-channels --channel defaults 2>/dev/null || true
    conda tos accept --override-channels \
      --channel https://repo.anaconda.com/pkgs/main \
      --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
    conda env create -f $BASE/seed42/environment.yaml -p $BASE/env

`conda tos accept` is needed because conda 25.x gates the defaults channel behind an interactive
prompt that would abort the build under `set -e`. Verify against the pins:

    $BASE/env/bin/python -c "import torch, pytorch_lightning as pl; print(torch.__version__, pl.__version__)"
    # expect  2.9.1+cu128 2.3.0

Budget 30–60 minutes for the solve and download on a shared filesystem.

---

## What changed on 2026-07-27

| | previous version | now |
|---|---|---|
| runtime | "40 training runs, each ~20 h on one L40S" | 20 h is the `--time` ceiling, not a runtime. 10 array tasks; measured 4 h 25 – 5 h 32 each in June. |
| concurrency | assumed all ten run together | **QOS `normal` caps the user at 4 GPUs.** Three waves. Steps 4 and 5 rewritten around it. |
| step 1 | rsync ~9 GB | ~98 MB top-up; the 8.4 GB pool is on scratch and verified byte-identical. Appendix keeps the full path. |
| step 2d | build the conda env | already built and verified; skip. Appendix keeps the build. |
| disk | not mentioned | 61 TB free, no quota; the collect step's 2× check passes trivially. Explained rather than worried about. |
| dereferencing | warned about in the abstract | recorded as having actually happened, with the 15 GB and 6,000 dangling links it produced. |
| checkpoints | not mentioned | collected since 2026-07-27; pull them, they are what makes the campaign repeatable. |
| step 5 | two post-seed checks | three — elapsed time added first, because the schedule depends on it and it is unmeasured. |
