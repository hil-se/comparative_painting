#!/bin/bash -l
#SBATCH --clusters=tigris
#SBATCH --partition=tigris
#SBATCH --account=rc-onboard
#SBATCH --job-name=art-rater-clip
#SBATCH --array=0-39
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --mail-user=slack:@xx4455
#SBATCH --mail-type=ALL
#SBATCH --output=/home/xx4455/paper-projects/logs/art-rater-clip-%A_%a.out
#SBATCH --error=/home/xx4455/paper-projects/logs/art-rater-clip-%A_%a.err

set -euo pipefail

spack env activate default-ml-aarch64-25050701
artifact_root=/home/xx4455/paper-projects/artifacts/comparative_painting

categories=(abstract abstract representational representational)
targets=(beauty liking beauty liking)
modes=(within cross)

case_index=$((SLURM_ARRAY_TASK_ID / 10))
mode_index=$(((SLURM_ARRAY_TASK_ID / 5) % 2))
rater=$((SLURM_ARRAY_TASK_ID % 5 + 1))
category="${categories[${case_index}]}"
target="${targets[${case_index}]}"
mode="${modes[${mode_index}]}"

output="${artifact_root}/results/rater/sidhu-${category}-${target}-${mode}-r${rater}-clip-vit-b32.csv"
checkpoints="${artifact_root}/checkpoints/rater/${category}-${target}-${mode}-r${rater}"

mkdir -p "${artifact_root}/results/rater" "${checkpoints}"
cd "${SLURM_SUBMIT_DIR}"

python code/extensions/run_sidhu_rater_extensions.py \
    --features "${artifact_root}/features/sidhu-clip-vit-b32.npz" \
    --data-dir Data \
    --category "${category}" \
    --target "${target}" \
    --mode "${mode}" \
    --rater "${rater}" \
    --objectives regression,hinge,bradley_terry \
    --n-values 1-10 \
    --seeds 0-9 \
    --epochs-regression 200 \
    --epochs-pairwise 100 \
    --batch-size 32 \
    --output "${output}" \
    --checkpoint-dir "${checkpoints}"
