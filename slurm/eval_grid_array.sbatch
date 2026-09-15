#!/usr/bin/env bash
#SBATCH --job-name=kys-raw-eval
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:<GPU_TYPE>:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=<TIME>
#SBATCH --array=0-35%8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# One array task = one raw-selected checkpoint (index into `python -m kys_eval.run_grid --list`).
#
# Before submitting, replace every <PLACEHOLDER> above (sbatch rejects them as-is):
#   <PARTITION>  a partition with H100 GPUs
#   <ACCOUNT>    your allocation account (delete the line if your cluster has none)
#   <GPU_TYPE>   the cluster's gres name for H100, e.g. h100 (or use --gres=gpu:1 plus a --constraint)
#   <TIME>       wall time per checkpoint; v2 took ~9 min on one H100 plus a ~3 GB download: 01:00:00
#
# Submit from the repository root (logs/ is relative to it):
#   mkdir -p logs
#   sbatch --export=ALL,KYS_EVAL_ROOT=$PWD slurm/eval_grid_array.sbatch                  # whole grid, 36 tasks
#   sbatch --array=0-11 --export=ALL,KYS_EVAL_ROOT=$PWD,KYS_GRID_ARGS="--seeds 42" \
#          slurm/eval_grid_array.sbatch                                                 # one seed, 12 tasks
# The --array range must match `python -m kys_eval.run_grid --list <same KYS_GRID_ARGS>`.
# Finished checkpoints are skipped, so resubmitting the same array is safe.
set -euo pipefail

: "${KYS_EVAL_ROOT:?export KYS_EVAL_ROOT=<path to the kys-eval checkout>}"
cd "$KYS_EVAL_ROOT"
# shellcheck disable=SC1091
source "${KYS_VENV:-$KYS_EVAL_ROOT/.venv}/bin/activate"
export HF_HOME="${HF_HOME:-$KYS_EVAL_ROOT/hf_cache}"

echo "[kys] host=$(hostname) array_task=${SLURM_ARRAY_TASK_ID} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
# shellcheck disable=SC2086  # KYS_GRID_ARGS is split into separate arguments on purpose
python -m kys_eval.run_grid --index "${SLURM_ARRAY_TASK_ID}" --delete-weights ${KYS_GRID_ARGS:-}
