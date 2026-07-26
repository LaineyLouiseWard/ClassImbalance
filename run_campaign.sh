#!/bin/bash
# ============================================================================
# Multi-seed campaign driver — ONE resumable command for the whole campaign.
#
#   bash run_campaign.sh            # run / resume the full 10-seed campaign
#   DRYRUN=1 bash run_campaign.sh   # print the exact plan, run nothing
#
# This is the LOCAL driver. The definitive 10-seed run is sonic/campaign/submit_campaign.sh;
# use this for local dev / single seeds.
#
# You do NOT need to remember which seed or stage you were on. Re-run this after
# ANY interruption (Ctrl-C, reboot, CUDA crash) and it picks up automatically:
#   - finished seeds are skipped (tracked in .campaign/seed_<N>.done),
#   - the in-progress seed resumes mid-stage from its last.ckpt (RESUME=1),
#   - remaining seeds then run, and finally the cross-seed aggregate is built.
#
# Run it inside tmux so it survives the terminal closing:
#   conda activate label-quality-ceiling
#   tmux new -s seeds          # detach: Ctrl-b then d   |   reattach: tmux attach -t seeds
#   bash run_campaign.sh
#
# To force ONE seed to re-run from scratch: delete .campaign/seed_<N>.done (and,
# for the root seed, .campaign/wiped) and its model_weights, then re-run.
# ============================================================================
set -euo pipefail

SEEDS=(42 43 44 45 46 47 48 49 50 51)   # LOCKED 10-seed campaign set
ROOT_SEED=42                      # the seed that lives in this repo (others = worktrees)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this repo = root seed
STATE="$ROOT/.campaign"
mkdir -p "$STATE"

# Pin worktrees to the root's current commit, DETACHED (not a branch): a branch can
# only be checked out in one worktree, and the seeds only ever write gitignored
# outputs, so they never need their own branch. Keep the code fixed during the run.
#
# Which is exactly why a dirty tree cannot be allowed to start: the root seed would run the
# working tree and the other nine would run HEAD, so one run of ten differs from the other nine
# and nothing records which. Refuse rather than warn.
DIRT="$(git -C "$ROOT" status --porcelain)"
if [ -n "$DIRT" ]; then
  echo "ERROR: the working tree is dirty. Commit or stash before launching:" >&2
  while IFS= read -r line; do echo "    $line" >&2; done <<<"$DIRT"
  echo "  Seed $ROOT_SEED would run these changes and seeds 43-51 would not." >&2
  exit 1
fi
CAMPAIGN_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"

# Fail fast if the conda env is not active (else the first seed dies hours in).
if ! python -c "import torch" >/dev/null 2>&1; then
  echo "ERROR: active 'python' has no PyTorch — run 'conda activate label-quality-ceiling' first."
  exit 1
fi

# Anything but empty or "0" is a dry run: DRYRUN=true used to submit for real.
DRY=0
case "${DRYRUN:-0}" in ""|0|no|false) DRY=0 ;; *) DRY=1 ;; esac

run() {   # echo the command; execute it unless DRYRUN is set
  echo "+ $*"
  [ "$DRY" = "1" ] || "$@"
}

for SEED in "${SEEDS[@]}"; do
  DONE="$STATE/seed_$SEED.done"
  if [ -f "$DONE" ]; then
    echo "================= seed $SEED: already complete, skipping ================="
    continue
  fi
  echo "========================= seed $SEED ========================="

  if [ "$SEED" = "$ROOT_SEED" ]; then
    DIR="$ROOT"
    # Clear orphaned old-named checkpoints ONCE, before the very first root run.
    if [ ! -f "$STATE/wiped" ]; then
      run rm -rf "$ROOT"/model_weights/biodiversity/*
      run touch "$STATE/wiped"
    fi
  else
    DIR="$(dirname "$ROOT")/$(basename "$ROOT")_seed$SEED"
    # Create the worktree + shared symlinks on first touch (all idempotent).
    [ -d "$DIR" ] || run git -C "$ROOT" worktree add --detach "$DIR" "$CAMPAIGN_COMMIT"
    # An EXISTING worktree is the hole the root-tree check does not cover: it was created at some
    # earlier commit and is never moved, so a resumed campaign can run seed 43 on different code
    # from seed 42 with nothing recording it. Check the commit and the cleanliness of each one.
    if [ -d "$DIR" ] && [ "$DRY" != "1" ]; then
      WT_COMMIT="$(git -C "$DIR" rev-parse HEAD)"
      WT_DIRT="$(git -C "$DIR" status --porcelain)"
      if [ "$WT_COMMIT" != "$CAMPAIGN_COMMIT" ]; then
        echo "ERROR: seed $SEED worktree $DIR is at $WT_COMMIT, campaign is at $CAMPAIGN_COMMIT." >&2
        echo "  Move it with: git -C '$DIR' checkout --detach $CAMPAIGN_COMMIT" >&2
        exit 1
      fi
      if [ -n "$WT_DIRT" ]; then
        echo "ERROR: seed $SEED worktree $DIR is dirty:" >&2
        while IFS= read -r line; do echo "    $line" >&2; done <<<"$WT_DIRT"
        exit 1
      fi
    fi
    run ln -sfn "$ROOT/data" "$DIR/data"
    # The WHOLE weights directory, not one file. pretrain_weights/ is gitignored, so a worktree gets
    # it empty, and this used to symlink only the teacher -- leaving stseg_base.pth absent. Three of
    # the four cells call ft_unetformer(pretrained=True, weight_path="pretrain_weights/stseg_base.pth")
    # at module scope, so they cannot even parse their config without it, and 36 of the 40 runs died.
    run rm -rf "$DIR/pretrain_weights"
    run ln -sfn "$ROOT/pretrain_weights" "$DIR/pretrain_weights"
  fi

  # The resumable per-seed run: sampler build, full leakage gate, training, evaluation (qualitative
  # are a separate once-off post-campaign step, not run per seed).
  run bash -c "cd '$DIR' && RESUME=1 SEED=$SEED HF_HUB_OFFLINE=1 \
      bash RUNBOOK.sh --from B4 --to C2"

  run touch "$DONE"
  echo "========================= seed $SEED: complete ========================="
done

echo "========================= aggregating all seeds ========================="
run bash -c "cd '$ROOT' && PYTHONPATH=. python scripts/analysis/aggregate_seeds.py --strict"
echo "Campaign complete → analysis/seed_aggregate/ (summary.json, report.md, ablation_*.tex, figure10_iou_*.csv)"
