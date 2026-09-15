#!/usr/bin/env bash
#SBATCH --job-name=kys-reference-check
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --gres=gpu:<GPU_TYPE>:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=<TIME>
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Re-score the v2 reference checkpoint (diversity_oriented seed42 ep3) and compare against
# reference/expected_diversity_oriented_seed42_ep3.json. Replace the <PLACEHOLDERS> as in
# slurm/eval_grid_array.sbatch (<TIME>: 01:00:00 is ample).
#
#   mkdir -p logs
#   sbatch --export=ALL,KYS_EVAL_ROOT=$PWD,KYS_REFERENCE_MODEL=/path/to/diversity-first-10B-1.5B-seed42/14305 \
#          slurm/reference_check.sbatch
# KYS_REFERENCE_MODEL may also be hf://<org>/<repo>/<subfolder>. The job exits 0 only on PASS.
set -euo pipefail

: "${KYS_EVAL_ROOT:?export KYS_EVAL_ROOT=<path to the kys-eval checkout>}"
: "${KYS_REFERENCE_MODEL:?export KYS_REFERENCE_MODEL=<reference checkpoint dir or hf:// spec>}"
cd "$KYS_EVAL_ROOT"
# shellcheck disable=SC1091
source "${KYS_VENV:-$KYS_EVAL_ROOT/.venv}/bin/activate"
export HF_HOME="${HF_HOME:-$KYS_EVAL_ROOT/hf_cache}"

echo "[kys] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python -m kys_eval.reference_check --model "$KYS_REFERENCE_MODEL"
