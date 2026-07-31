#!/bin/bash -l
#SBATCH --clusters=tigris
#SBATCH --partition=tigris
#SBATCH --account=rc-onboard
#SBATCH --job-name=art-apddv2-full
#SBATCH --array=0-21
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-user=slack:@xx4455
#SBATCH --mail-type=ALL
#SBATCH --output=/home/xx4455/paper-projects/logs/art-apddv2-full-%A_%a.out
#SBATCH --error=/home/xx4455/paper-projects/logs/art-apddv2-full-%A_%a.err

set -euo pipefail

spack env activate default-ml-aarch64-25050701
export TF_FORCE_GPU_ALLOW_GROWTH=true
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
slugs=(
    total
    theme-logic
    creativity
    layout-composition
    space-perspective
    order
    light-shadow
    color
    details-texture
    overall
    mood
)

target_index=$((SLURM_ARRAY_TASK_ID / 2))
if ((SLURM_ARRAY_TASK_ID % 2 == 0)); then
    representation=resnet50
else
    representation=clip-vit-b32
fi
target="${targets[${target_index}]}"
slug="${slugs[${target_index}]}"
manifest="${artifact_root}/manifests/apddv2.csv"
features="${artifact_root}/features/apddv2-${representation}.npz"
output="${artifact_root}/results/apddv2-${slug}-${representation}.csv"
checkpoints="${artifact_root}/checkpoints/apddv2-${slug}-${representation}"

mkdir -p "${artifact_root}/results" "${checkpoints}"
cd "${SLURM_SUBMIT_DIR}"

python code/extensions/run_art_extensions.py \
    --manifest "${manifest}" \
    --features "${features}" \
    --dataset apddv2 \
    --representation "${representation}" \
    --target "${target}" \
    --objectives regression,hinge,bradley_terry \
    --n-values 1-10 \
    --seeds 0-9 \
    --epochs-regression 200 \
    --epochs-pairwise 100 \
    --batch-size 256 \
    --output "${output}" \
    --checkpoint-dir "${checkpoints}"
