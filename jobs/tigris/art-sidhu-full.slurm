#!/bin/bash -l
#SBATCH --clusters=tigris
#SBATCH --partition=tigris
#SBATCH --account=rc-onboard
#SBATCH --job-name=art-sidhu-full
#SBATCH --array=0-7
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --mail-user=slack:@xx4455
#SBATCH --mail-type=ALL
#SBATCH --output=/home/xx4455/paper-projects/logs/art-sidhu-full-%A_%a.out
#SBATCH --error=/home/xx4455/paper-projects/logs/art-sidhu-full-%A_%a.err

set -euo pipefail

spack env activate default-ml-aarch64-25050701
artifact_root=/home/xx4455/paper-projects/artifacts/comparative_painting
categories=(abstract abstract abstract abstract representational representational representational representational)
targets=(beauty beauty liking liking beauty beauty liking liking)
representations=(resnet50 clip-vit-b32 resnet50 clip-vit-b32 resnet50 clip-vit-b32 resnet50 clip-vit-b32)

category="${categories[${SLURM_ARRAY_TASK_ID}]}"
target="${targets[${SLURM_ARRAY_TASK_ID}]}"
representation="${representations[${SLURM_ARRAY_TASK_ID}]}"
manifest="${artifact_root}/manifests/sidhu.csv"
features="${artifact_root}/features/sidhu-${representation}.npz"
output="${artifact_root}/results/sidhu-${category}-${target}-${representation}.csv"
checkpoints="${artifact_root}/checkpoints/sidhu-${category}-${target}-${representation}"

mkdir -p "${artifact_root}/results" "${checkpoints}"
cd "${SLURM_SUBMIT_DIR}"

python code/extensions/run_art_extensions.py \
    --manifest "${manifest}" \
    --features "${features}" \
    --dataset sidhu \
    --representation "${representation}" \
    --category "${category}" \
    --target "${target}" \
    --objectives regression,hinge,bradley_terry \
    --n-values 1-10 \
    --seeds 0-9 \
    --epochs-regression 200 \
    --epochs-pairwise 100 \
    --batch-size 32 \
    --output "${output}" \
    --checkpoint-dir "${checkpoints}"
