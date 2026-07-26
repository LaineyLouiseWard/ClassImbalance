#!/bin/bash
# ============================================================================
# Submit the 10-seed 2x2 factorial campaign to Slurm.
#
#   DRYRUN=1 bash sonic/campaign/submit_campaign.sh    # print the plan, submit nothing
#   bash sonic/campaign/submit_campaign.sh             # submit
#
# Everything cluster-specific comes from the environment, so this file names no person, no
# account and no home directory:
#
#   SONIC_USER      cluster username                    (default: $USER)
#   SONIC_SCRATCH   campaign root on scratch            (default: /home/people/$SONIC_USER/scratch/lqc)
#   SONIC_ACCOUNT   Slurm account                       (default: shared_acc)
#   SONIC_MAIL      address for END,FAIL mail           (optional; no mail if unset)
#   SPLIT_TAG       which split to run                  (default: f1)
#   BATCH_VARIANT   b2 (default) or b4
#
# It refuses to start on a dirty working tree. The array pins nine of the ten seeds to whatever
# the seed worktrees hold, so an uncommitted change would run for some seeds and not others and
# leave no record of which.
# ============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SONIC_USER="${SONIC_USER:-$USER}"
SONIC_SCRATCH="${SONIC_SCRATCH:-/home/people/${SONIC_USER}/scratch/lqc}"
SONIC_ACCOUNT="${SONIC_ACCOUNT:-shared_acc}"
SPLIT_TAG="${SPLIT_TAG:-f1}"
BATCH_VARIANT="${BATCH_VARIANT:-b2}"
DRYRUN="${DRYRUN:-0}"

# ---- refuse a dirty tree ---------------------------------------------------
if ! git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  echo "ERROR: $REPO is not a git checkout, so the campaign commit cannot be recorded." >&2
  exit 1
fi
DIRT="$(git -C "$REPO" status --porcelain)"
if [ -n "$DIRT" ]; then
  echo "ERROR: the working tree is dirty. Commit or stash before launching:" >&2
  while IFS= read -r line; do echo "    $line" >&2; done <<<"$DIRT"
  echo "  Forty runs are attributed to one commit; an uncommitted change makes that false." >&2
  exit 1
fi
COMMIT="$(git -C "$REPO" rev-parse HEAD)"

# ---- plan ------------------------------------------------------------------
LOGDIR="${SONIC_SCRATCH}/logs"
SEEDS="42..51"
CELLS="baseline, transfer-only, sampler-only, full"

cat <<PLAN
================================================================
 campaign plan
   repo            $REPO
   commit          $COMMIT (clean)
   scratch root    $SONIC_SCRATCH
   logs            $LOGDIR
   account         $SONIC_ACCOUNT
   split           $SPLIT_TAG
   batch variant   $BATCH_VARIANT
   seeds           $SEEDS  (10 array tasks)
   cells per seed  $CELLS  (4)
   total runs      40
   scored on       val, test (Test A), external_test (Test B)
================================================================
PLAN

SBATCH_ARGS=(
  --account="$SONIC_ACCOUNT"
  --output="${LOGDIR}/%x_seed%a_%A.out"
  --error="${LOGDIR}/%x_seed%a_%A.err"
  --export="ALL,SONIC_SCRATCH=${SONIC_SCRATCH},SPLIT_TAG=${SPLIT_TAG},BATCH_VARIANT=${BATCH_VARIANT}"
)
# A bare `[ ... ] && ...` here would return 1 when the test fails, which under `set -e` exits
# the script before it ever submits.
if [ -n "${SONIC_MAIL:-}" ]; then
  SBATCH_ARGS+=(--mail-type="END,FAIL" --mail-user="$SONIC_MAIL")
fi

if [ "$DRYRUN" = "1" ]; then
  echo "DRYRUN: would run"
  printf '  sbatch'
  printf ' %q' "${SBATCH_ARGS[@]}" "${REPO}/sonic/campaign/campaign.slurm"
  printf '\n'
  echo "DRYRUN: nothing submitted."
  exit 0
fi

mkdir -p "$LOGDIR"
sbatch "${SBATCH_ARGS[@]}" "${REPO}/sonic/campaign/campaign.slurm"
