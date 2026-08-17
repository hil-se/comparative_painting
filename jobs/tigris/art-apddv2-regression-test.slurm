#!/bin/bash -l
#SBATCH --clusters=sporc
#SBATCH --partition=onboard
#SBATCH --account=loop
#SBATCH --qos=qos_tier3
#SBATCH --job-name=art-apdd-reg-test
#SBATCH --array=0-10%8
#SBATCH --gres=shard:a100:32
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-user=slack:@xx4455
#SBATCH --mail-type=FAIL
#SBATCH --output=/home/xx4455/paper-projects/logs/art-apdd-reg-test-%A_%a.out
#SBATCH --error=/home/xx4455/paper-projects/logs/art-apdd-reg-test-%A_%a.err

set -euo pipefail

spack env activate default-ml-x86_64-25050601
export TF_GPU_MEMORY_LIMIT_MB=24576
export PYTHONUNBUFFERED=1
artifact_root=/home/xx4455/paper-projects/artifacts/comparative_painting
targets=(
    "Total aesthetic score"
    "Theme and logic"
    "Creativity"
    "Layout and composition"
    "Space and perspective"
    "The sense of order"
    "Light and shadow"
    "Color"
    "Details and texture"
    "The overall"
    "Mood"
)
slugs=(total theme-logic creativity layout-composition space-perspective order light-shadow color details-texture overall mood)

target="${targets[${SLURM_ARRAY_TASK_ID}]}"
slug="${slugs[${SLURM_ARRAY_TASK_ID}]}"
tuning_root="${artifact_root}/tuning/apddv2-regression"
output_root="${tuning_root}/test"
mkdir -p "${output_root}/raw"
cd "${SLURM_SUBMIT_DIR}"

python code/extensions/tune_apddv2_regression.py \
    --phase test \
    --manifest "${artifact_root}/manifests/apddv2.csv" \
    --features "${artifact_root}/features/apddv2-clip-vit-b32.npz" \
    --target "${target}" \
    --config-ids-file "${tuning_root}/confirm/summary/winner.json" \
    --seeds 0-9 \
    --epochs 200 \
    --patience 20 \
    --output "${output_root}/raw/${slug}.csv"
