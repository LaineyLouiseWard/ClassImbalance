# Brief — stage the campaign to Sonic and launch it

Paste everything below the rule into a **fresh chat**. Run it only after
`docs/BRIEF_FINAL_AUDIT.md` has come back and anything launch-blocking is fixed.

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

<what_you_are_staging>
`label-quality-ceiling`. Four factorial cells x ten seeds (42–51) = 40 training runs, each ~20 h on
one L40S. Two data-curation interventions on a fixed FT-UNetFormer; the paper's contribution is a
boundary-label diagnosis.

**The launcher is `sonic/campaign/submit_campaign.sh`.** It is the only supported path. Do NOT use
anything else under `sonic/` — those are the pre-rebuild scripts, they hard-code one person's student
number, and `sonic/10_submit_final_campaign.slurm` bypasses the preflight gate entirely and tests for
artefacts that were withdrawn. `sonic/*` is gitignored EXCEPT `sonic/campaign/`, which is the tracked
path and the one you want.

**What the launcher refuses to do**, so you know what it is checking rather than being surprised:
- it refuses on a dirty working tree, because nine of the ten seeds run a pinned commit
- it refuses if `$SONIC_SCRATCH/env` or any of `$SONIC_SCRATCH/seed42 … seed51` is missing
- each array task re-checks the commit it is actually running and aborts if it differs
- each array task runs the full leakage gate BEFORE training and aborts if it fails

**Environment variables it reads** (all optional except where noted):

    SONIC_USER      cluster username                  (default: $USER)
    SONIC_SCRATCH   campaign root on scratch          (default: /home/people/$SONIC_USER/scratch/lqc)
    SONIC_ACCOUNT   Slurm account                     (default: shared_acc)
    SONIC_MAIL      address for END,FAIL mail         (optional; no mail if unset)
    SPLIT_TAG       which split to run                (default: f1)
    BATCH_VARIANT   b2 = batch 2 / lr 3e-4 (default), b4 = batch 4 / lr 6e-4
</what_you_are_staging>

<the_data_layout_that_matters>
**Read this before you rsync anything. Getting it wrong is the single most likely way to waste a day.**

The campaign's data is a two-level chain of RELATIVE symlinks, not three independent copies:

    data/biodiversity_split/     8.4 GB   REAL FILES — the 2,143-tile pool
    data/openearthmap_relabelled/  69 MB  REAL FILES — the relabelled OEM tiles
    data/split_f1/                12 MB   SYMLINKS -> ../../../biodiversity_split/<split>/...
    data/oem_combined_f1/         17 MB   SYMLINKS -> ../../../split_f1/...  and
                                                   -> ../../../openearthmap_relabelled/...
    pretrain_weights/            542 MB   REAL FILES — stseg_base.pth (485 MB) + the teacher

So `data/oem_combined_f1` points at `data/split_f1`, which points at `data/biodiversity_split`.

Three consequences:

1. **Transfer ~9.0 GB of real files plus ~28 MB of symlink trees, not 13 GB.** Send
   `biodiversity_split`, `openearthmap_relabelled`, `split_f1`, `oem_combined_f1` and
   `pretrain_weights`.
2. **Preserve the symlinks. Do NOT use `rsync -L`.** Dereferencing turns 28 MB of links into ~13 GB
   of duplicated pixels, and breaks the guarantee that the pre-training pool and the training split
   are literally the same bytes — which is a property the leakage gate checks.
3. **The relative paths only resolve if all four directories sit under one common `data/` parent.**
   Stage them into a single `data_stage/data/` that mirrors the repo layout exactly, then symlink
   that one directory into each seed tree. Do not stage them separately.

`data/biodiversity_raw` (9 GB) and `data/openearthmap_raw` (25 GB) are **not** needed: they are
A-stage inputs and the A stages have already run. Do not send them.

**`notes/` must never travel.** It is gitignored, so a `git clone` will not carry it — which is one
reason the code arrives by clone rather than by rsync. If you rsync the repo for any reason, exclude
it explicitly.

**Every artefact the campaign needs is committed** and arrives with the clone: the split manifest, the
sampler weights, the augmentation list, the teacher confusion, the boundary denominators, the
correlogram. You do not need to transfer anything from `artifacts/`.
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

## 1. Transfer the data (~9 GB, resumable)

From the laptop. Set these once and reuse them:

    SONIC_USER=<username>            # ask the user; do not guess
    HOST=login.ucd.ie
    BASE=/home/people/$SONIC_USER/scratch/lqc
    STAGE=$BASE/data_stage

    ssh $SONIC_USER@$HOST "mkdir -p $STAGE/data $BASE/logs"

    # -a preserves symlinks as symlinks. NEVER add -L here.
    rsync -avh --partial --progress \
      data/biodiversity_split data/openearthmap_relabelled data/split_f1 data/oem_combined_f1 \
      "$SONIC_USER@$HOST:$STAGE/data/"

    rsync -avh --partial --progress pretrain_weights "$SONIC_USER@$HOST:$STAGE/"

`rsync` is resumable — if it drops, re-run the same command.

**Verify before moving on.** On Sonic:

    cd $STAGE/data
    for s in train val test external_test; do printf "%-14s %s\n" $s $(ls split_f1/$s/images | wc -l); done
    # expect  train 1072   val 173   test 294   external_test 191
    ls oem_combined_f1/train/images | wc -l        # expect 3190
    find split_f1 oem_combined_f1 -xtype l | wc -l # expect 0  (no dangling symlinks)
    ls -lh $STAGE/pretrain_weights/stseg_base.pth  # expect ~485 MB

A non-zero dangling count means the symlinks did not resolve — almost always because the four data
directories are not under one common parent. Fix the layout, do not dereference.

## 2. Build the environment and the ten seed trees, on the LOGIN node

The login node has internet; compute nodes do not. The env build needs internet, the campaign does
not (`HF_HUB_OFFLINE=1` is set at submit and every cell either loads a local checkpoint or a local
`.pth`).

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

    # 2d. conda env in scratch (home quota is small)
    module load anaconda3/2025.12-1
    conda tos accept --override-channels --channel defaults 2>/dev/null || true
    conda tos accept --override-channels \
      --channel https://repo.anaconda.com/pkgs/main \
      --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
    conda env create -f $BASE/seed42/environment.yaml -p $BASE/env

`conda tos accept` is needed because conda 25.x gates the defaults channel behind an interactive
prompt that would abort the build under `set -e`.

**Verify before moving on:**

    conda run -p $BASE/env python -c "import torch, pytorch_lightning as pl; print(torch.__version__, pl.__version__)"
    for S in 42 43 44 45 46 47 48 49 50 51; do
      printf "seed%-3s %s %s\n" $S "$(git -C $BASE/seed$S rev-parse --short HEAD)" \
        "$(ls $BASE/seed$S/data/split_f1/train/images 2>/dev/null | wc -l)"
    done
    # every row must show the SAME short commit and 1072

The repo uses `pytorch_lightning`, not the unified `lightning` namespace — import that one or a
healthy env will look broken.

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

If the gate fails here it will fail in all ten tasks. Fix the staging, not the gate.

## 4. Dry run, then submit

From the laptop, or from `$BASE/seed42` on Sonic:

    SONIC_USER=<username> SONIC_SCRATCH=$BASE DRYRUN=1 bash sonic/campaign/submit_campaign.sh

Read the plan it prints: commit, scratch root, account, split tag, batch variant, 10 tasks, 4 cells,
40 runs. Confirm the commit matches step 0. **If it refuses, it is telling you something real** —
a dirty tree, or a missing seed directory. Fix that; do not work around it.

Then, for real:

    SONIC_USER=<username> SONIC_SCRATCH=$BASE bash sonic/campaign/submit_campaign.sh
    squeue -u $SONIC_USER

## 5. Watch the first task, and STOP

Do not walk away with all ten running blind. Within the first hour:

    tail -f $BASE/logs/lqc-campaign_seed0_*.out

Confirm, in order: the provenance line naming the commit; `LEAKAGE PREFLIGHT PASSED`; `[B4]` building
the sampler; `[B4b]` passing; then `[B1]` starting to train. The gate runs before any training by
design — if you see `[B1]` before `[B4b]`, stop and report it.

**When seed 42 finishes, stop and check two things before the other nine complete:**

1. **Convergence.** The 45-epoch budget was set for 1,706 training tiles and there are now 1,072. Read
   the validation curve from `$BASE/seed42/lightning_logs/`. If `val_mIoU` is still climbing at epoch
   45, the baseline is under-trained and every intervention will look better than it is. That is a
   result worth knowing before spending the other 39 runs.
2. **Provenance.** `cat $BASE/seed42/model_weights/biodiversity/*/run_provenance_seed42.json` — commit,
   dirty flag, GPU, precision, torch and lightning versions. Confirm the commit is the campaign commit
   and `dirty` is `false`.

## 6. Fetch and aggregate

The array writes per-seed metrics to `$BASE/results/{val,test,external}/seed<N>_<cell>.json` and the
per-seed softmax dumps under each seed tree.

    rsync -avh "$SONIC_USER@$HOST:$BASE/results/" ./sonic/results_f1/

    SPLIT_TAG=f1 PYTHONPATH=. python scripts/analysis/aggregate_seeds.py \
      --results-dir          sonic/results_f1/val \
      --test-results-dir     sonic/results_f1/test \
      --external-results-dir sonic/results_f1/external \
      --strict

`--strict` fails if any of 4 cells x 3 splits x 10 seeds is missing. The aggregator also checks each
file's recorded checkpoint against the cell and split tag being asked for, so a metrics file from
another campaign is rejected rather than averaged in.
</procedure>

<constraints>
- Do NOT modify the repository. The commit that runs must be the commit that was audited.
- Do NOT use any script under `sonic/` except `sonic/campaign/`.
- Do NOT use `rsync -L` on the data.
- Do NOT transfer `notes/`, `data/biodiversity_raw` or `data/openearthmap_raw`.
- Ask the user for the cluster username and the Slurm account rather than guessing. Do not put either
  into a tracked file.
- If a gate refuses, report it and stop. Every refusal in this pipeline was added because the thing it
  refuses actually happened.
</constraints>

<output_format>
1. **Where staging got to**, step by step, with the verification output for each.
2. **Anything that refused**, what it said, and what you did.
3. **The submitted job id**, and the plan the dry run printed.
4. **The first task's first hour**: gate output, stage order, and whether training started.
5. **Anything you could not verify** from the laptop side.
</output_format>
