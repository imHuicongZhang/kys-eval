#!/usr/bin/env bash
#SBATCH --job-name=kys-raw-eval
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:<GPU_TYPE>:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=<TIME>
#SBATCH --array=0-26%8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# One array task = one raw-selected checkpoint (index into `python -m kys_eval.run_grid --list <KYS_GRID_ARGS>`).
#
# Before submitting, replace every <PLACEHOLDER> above (sbatch rejects them as-is):
#   <PARTITION>  a partition with H100 GPUs
#   <ACCOUNT>    your allocation account (delete the line if your cluster has none)
#   <GPU_TYPE>   the cluster's gres name for H100, e.g. h100 (or use --gres=gpu:1 plus a --constraint)
#   <TIME>       wall time per checkpoint; v2 took ~9 min on one H100 plus a ~3 GB download: 01:00:00
#
# KYS_GRID_ARGS is REQUIRED (there is no implicit selection): the run_grid filters that define the indices.
# Use KYS_GRID_ARGS=all for the full 72-cell grid. Every task also checks that the --array range is exactly
# 0..N-1 for the N cells the filters select, and refuses otherwise, so an array range and a filter that do
# not belong together cannot score the wrong cells (prelaunch audit E-F2).
#
# Submit from the repository root (logs/ is relative to it):
#   mkdir -p logs
#   R="--settings raw_diversity_oriented raw_disagreement_aware signal-disagreement-lambda0"
#   sbatch --array=0-26%8 --export=ALL,KYS_EVAL_ROOT=$PWD,KYS_GRID_ARGS="$R" slurm/eval_grid_array.sbatch  # rebuttal set, 27
#   sbatch --array=0-71%8 --export=ALL,KYS_EVAL_ROOT=$PWD,KYS_GRID_ARGS=all slurm/eval_grid_array.sbatch   # whole grid, 72
#   sbatch --array=0-23 --export=ALL,KYS_EVAL_ROOT=$PWD,KYS_GRID_ARGS="--seeds 42" \
#          slurm/eval_grid_array.sbatch                                                 # one seed, 24 tasks
# To re-run only some indices of a selection (e.g. --array=7,12), also export KYS_ARRAY_PARTIAL=1: the range
# check is then skipped, so make sure the indices come from `run_grid --list` with the same KYS_GRID_ARGS.
# Finished checkpoints are skipped (only if their results belong to that cell), so resubmitting is safe.
set -euo pipefail

: "${KYS_EVAL_ROOT:?export KYS_EVAL_ROOT=<path to the kys-eval checkout>}"
cd "$KYS_EVAL_ROOT"
# shellcheck disable=SC1091
source "${KYS_VENV:-$KYS_EVAL_ROOT/.venv}/bin/activate"
export HF_HOME="${HF_HOME:-$KYS_EVAL_ROOT/hf_cache}"

if [[ -z "${KYS_GRID_ARGS:-}" ]]; then
  echo "[kys] KYS_GRID_ARGS is not set; refusing. Export the run_grid filters that define the array indices," >&2
  echo "[kys] e.g. KYS_GRID_ARGS=\"--settings raw_diversity_oriented raw_disagreement_aware signal-disagreement-lambda0\"" >&2
  echo "[kys] (rebuttal set, --array=0-26), or KYS_GRID_ARGS=all for the 72-cell grid (--array=0-71)." >&2
  exit 2
fi
GRID_ARGS="$KYS_GRID_ARGS"
[[ "$GRID_ARGS" == all ]] && GRID_ARGS=""
EXPECT=()
if [[ "${KYS_ARRAY_PARTIAL:-0}" != 1 ]]; then
  if [[ "${SLURM_ARRAY_TASK_MIN:-}" != 0 || -z "${SLURM_ARRAY_TASK_MAX:-}" ]]; then
    echo "[kys] the --array range must start at 0 (got min=${SLURM_ARRAY_TASK_MIN:-unset});" \
         "export KYS_ARRAY_PARTIAL=1 to re-run selected indices" >&2
    exit 2
  fi
  EXPECT=(--expect-cells "$((SLURM_ARRAY_TASK_MAX + 1))")
fi

echo "[kys] host=$(hostname) array_task=${SLURM_ARRAY_TASK_ID} grid_args=${KYS_GRID_ARGS} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
# shellcheck disable=SC2086  # GRID_ARGS is split into separate arguments on purpose
python -m kys_eval.run_grid --index "${SLURM_ARRAY_TASK_ID}" "${EXPECT[@]}" --delete-weights ${GRID_ARGS}
